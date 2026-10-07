"""Section x section similarity heatmap for one labeled (JD, CV) pair.

Usage:
    python scripts/heatmap.py
    python scripts/heatmap.py --jd-id jd_0001 --cv-id cv_0003
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from cvjd.io import load_documents, load_queries  # noqa: E402
from cvjd.models.embedding import create_embedder  # noqa: E402
from cvjd.models.matching import cosine, embed_corpus  # noqa: E402


def pick_pair(queries, jd_id: str | None, cv_id: str | None) -> tuple[str, str]:
    if jd_id and cv_id:
        return jd_id, cv_id
    for query in queries:
        label_by_cv = dict(zip(query.cv_ids, query.relevance))
        for candidate, label in label_by_cv.items():
            if label == 3:
                return query.jd_id, candidate
    for query in queries:
        for candidate, label in zip(query.cv_ids, query.relevance):
            if label >= 0:
                return query.jd_id, candidate
    raise SystemExit("no labeled pair found")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--embedder", default="mock", choices=["mock", "st"])
    parser.add_argument("--jd-id", default=None)
    parser.add_argument("--cv-id", default=None)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results")
    args = parser.parse_args(argv)

    cvs = load_documents(args.data_dir / "raw" / "cvs.jsonl")
    jds = load_documents(args.data_dir / "raw" / "jds.jsonl")
    queries = load_queries(args.data_dir / "labels" / "queries.jsonl")

    jd_id, cv_id = pick_pair(queries, args.jd_id, args.cv_id)
    jd = jds[jd_id]
    cv = cvs[cv_id]

    embedder = create_embedder(args.embedder)
    jd_emb = embed_corpus([jd], embedder)[jd_id]
    cv_emb = embed_corpus([cv], embedder)[cv_id]

    jd_names = sorted(jd.sections, key=lambda n: jd.sections[n].order)
    cv_names = sorted(cv.sections, key=lambda n: cv.sections[n].order)
    matrix = np.array([
        [cosine(jd_emb[name], cv_emb[other]) for other in cv_names]
        for name in jd_names
    ])

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, vmin=0.0, vmax=max(1.0, float(matrix.max())), cmap="viridis")
    ax.set_xticks(range(len(cv_names)), cv_names, rotation=45, ha="right")
    ax.set_yticks(range(len(jd_names)), jd_names)
    ax.set_xlabel(f"CV sections ({cv_id})")
    ax.set_ylabel(f"JD sections ({jd_id})")
    ax.set_title("Section-to-section cosine similarity")
    for i in range(len(jd_names)):
        for j in range(len(cv_names)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                    color="white" if matrix[i, j] < 0.5 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"heatmap_{jd_id}_{cv_id}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
