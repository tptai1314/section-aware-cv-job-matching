"""Tests for embedding compression and two-stage retrieval (RQ3)."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cvjd.models.acceleration import (  # noqa: E402
    dequantize_int8,
    fit_pca,
    prepare_compressed_single,
    prepare_two_stage,
    quantize_int8,
)
from cvjd.models.embedding import MockEmbedder  # noqa: E402
from cvjd.schema import Document, Section  # noqa: E402


def _doc(doc_id: str, doc_type: str, skills: str) -> Document:
    section = Section("skills", skills, "SKILLS", 0)
    return Document(
        doc_id=doc_id, doc_type=doc_type, lang="en",
        raw_text=f"SKILLS\n{skills}", sections={"skills": section},
    )


def _corpus():
    jds = [_doc("jd_1", "jd", "python kubernetes")]
    cvs = [
        _doc("cv_1", "cv", "python kubernetes docker"),
        _doc("cv_2", "cv", "cooking recipes"),
        _doc("cv_3", "cv", "java spring"),
    ]
    return jds, cvs


def test_quantize_int8_round_trip_error_bound():
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(16, 384)).astype(np.float32)
    codes, scales = quantize_int8(vectors)
    assert codes.dtype == np.int8
    assert codes.shape == vectors.shape
    restored = dequantize_int8(codes, scales)
    per_row_scale = np.max(np.abs(vectors), axis=1) / 127.0
    error = np.max(np.abs(restored - vectors), axis=1)
    assert np.all(error <= per_row_scale / 2.0 + 1e-6)


def test_quantize_int8_zero_vector():
    codes, scales = quantize_int8(np.zeros((2, 8), dtype=np.float32))
    assert np.all(codes == 0)
    assert np.all(scales == 1.0)
    assert np.all(dequantize_int8(codes, scales) == 0.0)


def test_quantize_rejects_non_2d():
    with pytest.raises(ValueError):
        quantize_int8(np.zeros(8))


def test_fit_pca_output_shape():
    rng = np.random.default_rng(1)
    vectors = rng.normal(size=(20, 64))
    _, projected = fit_pca(vectors, 8)
    assert projected.shape == (20, 8)
    _, projected_small = fit_pca(vectors, 1000)
    assert projected_small.shape[0] == 20
    assert projected_small.shape[1] <= 20


def test_compressed_single_ranks_matching_cv_first():
    jds, cvs = _corpus()
    embedder = MockEmbedder(dim=128)
    for compression in ("none", "pca", "int8"):
        rank_one = prepare_compressed_single(
            jds, cvs, embedder, compression=compression, pca_dim=4
        )
        ranking = rank_one("jd_1")
        assert {cv_id for cv_id, _ in ranking} == {"cv_1", "cv_2", "cv_3"}
        assert ranking[0][0] == "cv_1"
        assert all(ranking[0][1] >= score for _, score in ranking)


def test_two_stage_returns_all_candidates_in_order():
    jds, cvs = _corpus()
    embedder = MockEmbedder(dim=128)
    rank_one = prepare_two_stage(jds, cvs, embedder, top_k=2)
    ranking = rank_one("jd_1")
    assert len(ranking) == 3
    assert {cv_id for cv_id, _ in ranking} == {"cv_1", "cv_2", "cv_3"}
    assert ranking[0][0] == "cv_1"


def test_two_stage_head_is_first_stage_head_tail_keeps_stage1_order():
    jds, cvs = _corpus()
    embedder = MockEmbedder(dim=128)
    stage1 = prepare_compressed_single(jds, cvs, embedder, compression="none")
    first = stage1("jd_1")
    rank_one = prepare_two_stage(jds, cvs, embedder, top_k=2)
    ranking = rank_one("jd_1")
    assert {cv_id for cv_id, _ in ranking[:2]} == {cv_id for cv_id, _ in first[:2]}
    assert [cv_id for cv_id, _ in ranking[2:]] == [cv_id for cv_id, _ in first[2:]]


def test_two_stage_with_pca_stage1():
    jds, cvs = _corpus()
    embedder = MockEmbedder(dim=128)
    rank_one = prepare_two_stage(
        jds, cvs, embedder, top_k=3, stage1_compression="pca", pca_dim=4
    )
    ranking = rank_one("jd_1")
    assert len(ranking) == 3
    assert ranking[0][0] == "cv_1"


def test_two_stage_rejects_bad_top_k():
    jds, cvs = _corpus()
    embedder = MockEmbedder(dim=128)
    with pytest.raises(ValueError):
        prepare_two_stage(jds, cvs, embedder, top_k=0)
