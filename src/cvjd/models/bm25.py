"""BM25 lexical baseline over full document text."""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Sequence, Tuple

from rank_bm25 import BM25Okapi

try:
    from pyvi import ViTokenizer
except ImportError:
    ViTokenizer = None

from ..schema import Document

_TOKEN_RE = re.compile(
    r"[a-z0-9àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ_]+",
    re.I,
)


def tokenize(text: str) -> List[str]:
    if not text:
        return []
    norm_text = unicodedata.normalize("NFC", text)
    if ViTokenizer is not None:
        try:
            norm_text = ViTokenizer.tokenize(norm_text)
        except Exception:
            pass
    return _TOKEN_RE.findall(norm_text.lower())


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
