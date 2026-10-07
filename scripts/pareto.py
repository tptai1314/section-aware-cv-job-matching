"""Pareto chart: quality (NDCG@10) vs latency from benchmark/rerank CSVs.

Usage: python scripts/pareto.py
Reads results/benchmark.csv (required) and results/rerank.csv (optional).
Writes results/pareto.png.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


def read_points(path: Path, label_field: str) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    latest: Dict[str, Dict[str, str]] = {}
    for row in rows:
        latest[row.get(label_field) or row.get("method", "")] = row
    return list(latest.values())


def pareto_front(points: List[Tuple[float, float]]) -> List[int]:
    """Indices of points not dominated on (latency, ndcg)."""
    front = []
    for i, (x, y) in enumerate(points):
        dominated = any(
            j != i and px <= x and py >= y and (px < x or py > y)
            for j, (px, py) in enumerate(points)
        )
        if not dominated:
            front.append(i)
    return sorted(front, key=lambda i: points[i][0])


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    benchmark_path = args.results_dir / "benchmark.csv"
    if not benchmark_path.exists():
        raise SystemExit(f"missing {benchmark_path}; run scripts/benchmark.py first")

    labeled: List[Tuple[str, float, float]] = []
    for row in read_points(benchmark_path, "method"):
        if row.get("method") == "two_stage":
            label = f"two_stage({row.get('params', '')})"
        else:
            label = row["method"]
        labeled.append((label, float(row["query_ms_mean"]), float(row[f"ndcg@{args.k}"])))

    for row in read_points(args.results_dir / "rerank.csv", "mode"):
        if row["mode"] == "full":
            label = "cross_encoder_full"
        else:
            label = f"ce_two_stage({row.get('first_stage', '')},k={row.get('top_k', '')})"
        labeled.append((label, float(row["query_ms_mean"]), float(row[f"ndcg@{args.k}"])))

    if not labeled:
        raise SystemExit("no points to plot")

    coords = [(x, y) for _, x, y in labeled]
    front_idx = set(pareto_front(coords))

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for i, (label, x, y) in enumerate(labeled):
        marker = "D" if i in front_idx else "o"
        color = "tab:red" if i in front_idx else "tab:blue"
        ax.scatter(x, y, marker=marker, color=color, s=70, zorder=3)
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8)

    front_points = sorted((coords[i] for i in front_idx), key=lambda p: p[0])
    if len(front_points) > 1:
        ax.plot(
            [p[0] for p in front_points],
            [p[1] for p in front_points],
            color="tab:red", linestyle="--", linewidth=1, alpha=0.7, label="Pareto front",
        )

    ax.set_xscale("log")
    ax.set_xlabel("mean query latency (ms, log scale)")
    ax.set_ylabel(f"NDCG@{args.k}")
    ax.set_title("Quality vs latency trade-off")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    out = args.out or (args.results_dir / "pareto.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"pareto front: {[labeled[i][0] for i in sorted(front_idx, key=lambda i: coords[i][0])]}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
