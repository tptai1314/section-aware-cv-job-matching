"""BM25 lexical baseline over full document text."""

from __future__ import annotations

import re
from typing import Dict, List, Sequence, Tuple

from rank_bm25 import BM25Okapi

from ..schema import Document

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Ranker:
    """Indexes CVs once; each JD query returns CVs sorted by BM25 score."""

    def __init__(self, cvs: Sequence[Document]) -> None:
        if not cvs:
            raise ValueError("BM25Ranker needs a non-empty CV corpus")
        self.cv_ids = [cv.doc_id for cv in cvs]
        self.bm25 = BM25Okapi([tokenize(cv.raw_text) for cv in cvs])

    def rank(self, jd: Document) -> List[Tuple[str, float]]:
        scores = self.bm25.get_scores(tokenize(jd.raw_text))
        scored = zip(self.cv_ids, (float(s) for s in scores))
        return sorted(scored, key=lambda item: (-item[1], item[0]))


def rank_all_bm25(
    jds: Sequence[Document], cvs: Sequence[Document]
) -> Dict[str, List[Tuple[str, float]]]:
    ranker = BM25Ranker(cvs)
    return {jd.doc_id: ranker.rank(jd) for jd in jds}
