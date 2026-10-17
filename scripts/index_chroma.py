"""Index dữ liệu thật vào ChromaDB (Giai đoạn B — plan §6).

- Tạo 2 collection `frevia_cvs`, `frevia_jobs` (cosine).
- Embed tất cả text (mỗi section + `_full`) trong MỘT lần gọi để tránh overhead
  batch nhỏ trên CPU.
- Mỗi tài liệu → vector mỗi section + vector full-text `_full`, ID ổn định
  `cv:{id}:v1:{section}` / `jd:{id}:v1:{section}`, metadata `document_id`,
  `document_version`, `section`, `status`, `language`, `embedding_model`,
  `content_hash`.
- Smoke query: embedding full-text câu hỏi mẫu, tìm top-5 job gần nhất theo cosine.

Usage:
    python scripts/index_chroma.py
    python scripts/index_chroma.py --clear --limit 20 --embedder mock
    python scripts/index_chroma.py --chroma data/chroma_dev
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from cvjd.data.loader import load_raw_dir  # noqa: E402
from cvjd.models.embedding import create_embedder  # noqa: E402
from cvjd.storage import chroma  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default=str(REPO_ROOT / "data" / "raw"))
    parser.add_argument("--chroma-dir", default=str(REPO_ROOT / "data" / "chroma"))
    parser.add_argument("--embedder", default="bge-m3", choices=["mock", "st", "bge-m3"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--clear", action="store_true", help="xóa 2 collection rồi tạo lại")
    parser.add_argument("--limit", type=int, default=0, help="giới hạn số tài liệu/loại (smoke)")
    args = parser.parse_args()

    cvs, jds = load_raw_dir(Path(args.raw_dir))
    if args.limit > 0:
        cvs, jds = cvs[: args.limit], jds[: args.limit]
    print(f"loaded {len(cvs)} cvs, {len(jds)} jds from {args.raw_dir}", flush=True)

    embedder = create_embedder(args.embedder, device=args.device)
    print(f"embedder: {args.embedder} dim={embedder.dim}")

    client = chroma.build_client(args.chroma_dir if args.chroma_dir else None)
    names = chroma.ensure_collections(client)
    if args.clear:
        from chromadb.errors import NotFoundError

        try:
            for name in names.values():
                client.delete_collection(name)
            names = chroma.ensure_collections(client)
            print("cleared and recreated collections")
        except NotFoundError:
            pass

    model_id = getattr(embedder, "model_id", None) or "mock"

    entries = []  # (doc_type, doc_id, doc_lang, section, text)
    for doc_type, docs in (("cv", cvs), ("jd", jds)):
        for doc in docs:
            for section, text in chroma.section_vectors(doc.sections, doc.raw_text).items():
                entries.append((doc_type, doc.doc_id, doc.lang, section, text))
    print(
        f"embedding {len(entries)} section/full vectors in one batch on {args.device} ...",
        flush=True,
    )
    vectors = embedder.encode([e[4] for e in entries])
    print("done.", flush=True)

    by_doc: dict = defaultdict(list)
    for (doc_type, doc_id, lang, section, text), vec in zip(entries, vectors):
        by_doc[(doc_type, doc_id, lang)].append((section, text, vec))

    stats = {"upserted": 0, "deleted_stale": 0, "deleted_leftover": 0}
    for (doc_type, doc_id, lang), records in by_doc.items():
        deltas = chroma.upsert_vectors(
            client,
            doc_type=doc_type,
            doc_id=doc_id,
            records=records,
            embed_model=model_id,
            version=1,
            language=lang,
            status="READY",
        )
        for key in stats:
            stats[key] += deltas[key]

    print(
        f"total upserted={stats['upserted']} stale_deleted={stats['deleted_stale']} "
        f"leftover_deleted={stats['deleted_leftover']}"
    )
    print(
        f"vectors: cv={chroma.count_vectors(client, 'cv')} jd={chroma.count_vectors(client, 'jd')}"
    )

    sample = "Freelancer fullstack React developer với Node.js, TypeScript và PostgreSQL"
    query_vec = embedder.encode([sample])
    hits = chroma.query(client, "jd", query_vec, n_results=5, status="READY")
    print("\nSMOKE: top-5 jobs gần nhất với câu hỏi mẫu (distance):")
    for hit in hits:
        md = hit["metadata"]
        print(f"  {hit['id']:35s} d={hit['distance']:.4f}  section={md.get('section')}")

    if hasattr(client, "close"):
        client.close()  # để process/sqlite-release; build_client ở process khác heal qua warm-up
    return 0


if __name__ == "__main__":
    raise SystemExit(main())