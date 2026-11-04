"""Chạy recommend cho TOÀN BỘ corpus cả hai chiều (jd2cv + cv2jd).

Xác định trước độ đầy đủ của Chroma index (thiếu doc nào thì index phần đó),
rồi chạy `MatchingService.recommend` cho từng query, ghi kết quả vào JSONL:

    {"direction": "jd2cv", "query_id": "jd_1", "results": [{"doc_id": ..., "score": ...}, ...]}

Usage:
    python scripts/recommend_all.py                       # top-5, cả 379 query
    python scripts/recommend_all.py --top-k 10 --keyword
    python scripts/recommend_all.py --direction jd2cv --limit 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):  # Windows console (GBK) không in được UTF-8
    sys.stdout.reconfigure(encoding="utf-8")

from cvjd.io import load_documents  # noqa: E402
from cvjd.models.bm25 import BM25Ranker  # noqa: E402
from cvjd.models.embedding import create_embedder  # noqa: E402
from cvjd.service import MatchingService  # noqa: E402
from cvjd.storage import chroma  # noqa: E402
from cvjd.utils.config import load_config  # noqa: E402


def _docs_needing_index(client, doc_type: str, docs: dict, embed_model: str) -> list[str]:
    """Doc_id chưa có vector, hoặc vector cũ của embedder khác (phải re-embed)."""
    stale = []
    for doc_id in docs:
        rows = chroma.get_document(client, doc_type, doc_id, version=None, include=["metadatas"])
        if not rows:
            stale.append(doc_id)
            continue
        models = {(r["metadata"] or {}).get("embedding_model") for r in rows}
        if models != {embed_model}:
            stale.append(doc_id)
    return stale


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--chroma-dir", type=Path, default=REPO_ROOT / "data" / "chroma")
    parser.add_argument("--direction", choices=("jd2cv", "cv2jd", "both"), default="both")
    parser.add_argument("--embedder", default="bge-m3")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--pool-size", type=int, default=None)
    parser.add_argument("--keyword", action="store_true", help="fuse thêm điểm BM25")
    parser.add_argument("--keyword-alpha", type=float, default=None)
    parser.add_argument("--limit", type=int, default=None, help="chỉ chạy N query đầu (test)")
    parser.add_argument("--resume", action="store_true", help="chạy tiếp file output cũ, bỏ query đã xong")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "recommend_all.jsonl")
    args = parser.parse_args(argv)

    started = time.time()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    cvs = load_documents(args.data_dir / "raw" / "cvs.jsonl")
    jds = load_documents(args.data_dir / "raw" / "jds.jsonl")
    documents = {**cvs, **jds}
    print(f"corpus: {len(cvs)} CVs, {len(jds)} JDs", flush=True)

    config = load_config()
    embedder = create_embedder(args.embedder)
    embed_model = getattr(embedder, "model_id", args.embedder)
    client = chroma.build_client(str(args.chroma_dir))
    chroma.ensure_collections(client)

    # Index phần còn thiếu/stale (không re-embed doc đã có đúng embedder).
    missing_cv = _docs_needing_index(client, "cv", cvs, embed_model)
    missing_jd = _docs_needing_index(client, "jd", jds, embed_model)
    to_index = [documents[d] for d in missing_cv + missing_jd]
    if to_index:
        print(f"index thêm {len(to_index)} docs chưa có vector...", flush=True)
        MatchingService(client, embedder, embed_model=embed_model, config=config).index(to_index)
    else:
        print("index đã đầy đủ, bỏ qua embed", flush=True)

    service = MatchingService(client, embedder, embed_model=embed_model, config=config)

    plans: list[tuple[str, dict, dict, str]] = []
    if args.direction in ("jd2cv", "both"):
        plans.append(("jd2cv", jds, cvs, "cv"))
    if args.direction in ("cv2jd", "both"):
        plans.append(("cv2jd", cvs, jds, "jd"))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Resume: đọc các query đã ghi để tránh chạy lại (process cũ có thể bị kill).
    done: set[tuple[str, str]] = set()
    mode = "w"
    if args.resume and args.out.exists():
        with args.out.open(encoding="utf-8") as fh_old:
            for line in fh_old:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                done.add((row["direction"], row["query_id"]))
        mode = "a"
        print(f"resume: bỏ qua {len(done)} query đã chạy trước đó", flush=True)

    perf_path = args.out.with_suffix(".perf.jsonl")  # log tốc độ từng query
    n_queries = 0
    n_skipped = 0
    t_run = time.perf_counter()
    with (
        args.out.open(mode, encoding="utf-8") as fh,
        perf_path.open(mode, encoding="utf-8") as fh_perf,
    ):
        for direction, queries, candidates, target_type in plans:
            ranker = BM25Ranker(list(candidates.values())) if args.keyword else None
            ids = sorted(queries)
            if args.limit:
                ids = ids[: args.limit]
            todo = [q for q in ids if (direction, q) not in done]
            n_skipped += len(ids) - len(todo)
            print(f"[{direction}] còn phải chạy {len(todo)}/{len(ids)} query", flush=True)
            for i, query_id in enumerate(todo, 1):
                t0 = time.perf_counter()
                keyword_scores = None
                if ranker is not None:
                    keyword_scores = dict(ranker.rank(queries[query_id]))
                recs = service.recommend(
                    queries[query_id],
                    target_type,
                    documents=candidates,
                    pool_size=args.pool_size,
                    top_k=args.top_k,
                    keyword_scores=keyword_scores,
                    keyword_alpha=args.keyword_alpha,
                )
                sec = time.perf_counter() - t0
                fh.write(json.dumps(
                    {
                        "direction": direction,
                        "query_id": query_id,
                        "results": [r.to_dict() for r in recs],
                    },
                    ensure_ascii=False,
                ) + "\n")
                fh_perf.write(json.dumps(
                    {"direction": direction, "query_id": query_id, "sec": round(sec, 2)},
                    ensure_ascii=False,
                ) + "\n")
                fh.flush()
                fh_perf.flush()  # crash-safe: mỗi query là 1 dòng vĩnh viễn
                n_queries += 1
                if i % 20 == 0 or i == len(todo):
                    rate = i / max(time.perf_counter() - t_run, 1e-9)
                    print(
                        f"[{direction}] {i}/{len(todo)} queries "
                        f"({rate:.2f} q/s, query cuối {sec:.1f}s)",
                        flush=True,
                    )
    elapsed = time.time() - started
    summary = {
        "ran_at_utc": stamp,
        "embedder": args.embedder,
        "top_k": args.top_k,
        "keyword": args.keyword,
        "queries_ran_now": n_queries,
        "queries_skipped": n_skipped,
        "queries_total_in_file": n_queries + n_skipped,
        "elapsed_sec": round(elapsed, 1),
        "avg_sec_per_query": round(elapsed / n_queries, 2) if n_queries else None,
        "out": str(args.out),
        "perf": str(perf_path),
    }
    summary_path = args.out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
