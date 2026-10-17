"""End-to-end recommendation CLI (plan §10.2, stages 4-6 glue).

Nạp corpus từ `data/raw`, index vào Chroma, rồi chạy `MatchingService` cho một
truy vấn bất kỳ theo hai chiều:

    JD -> CV   (freelancer/browse)
    CV -> JD   (recommended jobs)

Usage:
    python scripts/recommend.py --direction jd2cv --query jd_12 --top-k 5
    python scripts/recommend.py --direction cv2jd --query cv_3 --keyword
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from cvjd.io import load_documents  # noqa: E402
from cvjd.models.bm25 import BM25Ranker  # noqa: E402
from cvjd.models.embedding import create_embedder  # noqa: E402
from cvjd.service import MatchingService  # noqa: E402
from cvjd.storage import chroma  # noqa: E402
from cvjd.utils.config import load_config  # noqa: E402


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--direction", choices=("jd2cv", "cv2jd"), default="jd2cv")
    parser.add_argument("--query", required=True, help="doc_id của JD (jd2cv) hoặc CV (cv2jd)")
    parser.add_argument("--embedder", default="mock")
    parser.add_argument("--pool-size", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--keyword", action="store_true", help="fuse thêm điểm BM25")
    parser.add_argument("--keyword-alpha", type=float, default=None)
    args = parser.parse_args(argv)

    cvs = load_documents(args.data_dir / "raw" / "cvs.jsonl")
    jds = load_documents(args.data_dir / "raw" / "jds.jsonl")
    documents = {**cvs, **jds}
    if args.query not in documents:
        raise SystemExit(f"query {args.query!r} không có trong corpus")

    config = load_config()
    embedder = create_embedder(args.embedder)
    client = chroma.build_client(None)
    chroma.ensure_collections(client)
    service = MatchingService(
        client,
        embedder,
        embed_model=getattr(embedder, "model_id", args.embedder),
        config=config,
    )
    service.index(list(documents.values()))

    target_type = "cv" if args.direction == "jd2cv" else "jd"
    candidates = cvs if target_type == "cv" else jds

    keyword_scores = None
    if args.keyword:
        keyword_scores = dict(BM25Ranker(list(candidates.values())).rank(documents[args.query]))

    recommendations = service.recommend(
        documents[args.query],
        target_type,
        documents=candidates,
        pool_size=args.pool_size,
        top_k=args.top_k,
        keyword_scores=keyword_scores,
        keyword_alpha=args.keyword_alpha,
    )
    print(json.dumps([r.to_dict() for r in recommendations], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
