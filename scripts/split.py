"""Re-split raw documents with the rule-based splitter and report accuracy.

Usage: python scripts/split.py [--data-dir data]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from cvjd.data.preprocessing import resplit_document  # noqa: E402
from cvjd.io import load_documents, save_documents  # noqa: E402
from cvjd.schema import Document  # noqa: E402


def compare(original: Document, resplit: Document) -> Tuple[bool, str]:
    """Exact match requires same section names, order and text."""
    orig = [original.sections[name] for name in sorted(
        original.sections, key=lambda n: original.sections[n].order
    )]
    new = [resplit.sections[name] for name in sorted(
        resplit.sections, key=lambda n: resplit.sections[n].order
    )]
    if [s.name for s in orig] != [s.name for s in new]:
        return False, (
            f"names {[s.name for s in orig]} -> {[s.name for s in new]}"
        )
    for a, b in zip(orig, new):
        if a.text.strip() != b.text.strip():
            return False, f"text mismatch in section {a.name!r}"
    return True, ""


def run_split(docs: Sequence[Document]) -> Tuple[List[Document], List[str]]:
    resplit_docs: List[Document] = []
    failures: List[str] = []
    for doc in docs:
        rebuilt = resplit_document(doc)
        ok, reason = compare(doc, rebuilt)
        if not ok:
            failures.append(f"{doc.doc_id}: {reason}")
        resplit_docs.append(rebuilt)
    return resplit_docs, failures


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    args = parser.parse_args(argv)

    total_docs = 0
    total_failures: List[str] = []
    for name in ("cvs", "jds"):
        docs = list(load_documents(args.data_dir / "raw" / f"{name}.jsonl").values())
        rebuilt, failures = run_split(docs)
        save_documents(args.data_dir / "interim" / f"{name}.jsonl", rebuilt)
        total_docs += len(docs)
        total_failures.extend(failures)
        print(f"{name}: {len(docs)} docs, {len(failures)} mismatches")

    if total_failures:
        print("mismatch details (first 10):")
        for line in total_failures[:10]:
            print(f"  {line}")
    exact = total_docs - len(total_failures)
    print(f"exact match: {exact}/{total_docs}")
    print(f"wrote interim -> {args.data_dir / 'interim'}")


if __name__ == "__main__":
    main()
