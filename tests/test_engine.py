"""Tests HybridRanker (plan §7): fusion, chuẩn hoá và tổng hợp điểm."""

from cvjd.ranking import (
    CandidateProfile,
    HybridRanker,
    RankedCandidate,
    clamp01,
    min_max,
)
from cvjd.ranking.hybrid import JobRequirement
from cvjd.utils.config import get_ranking_weights


def test_clamp01_bounds():
    assert clamp01(-0.5) == 0.0
    assert clamp01(0.5) == 0.5
    assert clamp01(1.5) == 1.0


def test_min_max_normalizes_and_handles_constant():
    assert min_max([0.0, 1.0, 2.0]) == [0.0, 0.5, 1.0]
    assert min_max([3.0, 3.0, 3.0]) == [1.0, 1.0, 1.0]
    assert min_max([]) == []


def test_ranker_loads_weights_from_config():
    ranker = HybridRanker(config={})
    assert ranker.weights == get_ranking_weights()
    assert abs(sum(ranker.weights.values()) - 1.0) < 1e-9


def test_fuse_without_keyword_clamps_semantic():
    result = HybridRanker().fuse({"a": 0.4, "b": 1.3})
    assert result == {"a": 0.4, "b": 1.0}


def test_fuse_alpha_extremes():
    sem = {"a": 1.0, "b": 0.0}
    kw = {"a": 0.0, "b": 1.0}
    ranker = HybridRanker()
    only_sem = ranker.fuse(sem, kw, alpha=1.0)
    only_kw = ranker.fuse(sem, kw, alpha=0.0)
    assert only_sem["a"] == 1.0 and only_sem["b"] == 0.0
    assert only_kw["a"] == 0.0 and only_kw["b"] == 1.0


def test_fuse_covers_union_of_ids():
    result = HybridRanker().fuse({"a": 1.0}, {"b": 1.0}, alpha=0.5)
    assert set(result) == {"a", "b"}
    assert result["a"] == 0.5  # a chỉ có semantic, b chỉ có keyword
    assert result["b"] == 0.5


def test_score_is_weighted_sum_of_components():
    requirement = JobRequirement(semantic_base=1.0)
    ranker = HybridRanker()
    candidate = ranker.score(requirement, CandidateProfile(), 0.25)
    expected = sum(
        ranker.weights[name] * value
        for name, value in candidate.as_breakdown().items()
    )
    assert abs(candidate.score - expected) < 1e-9
    assert candidate.semantic == 0.25
    assert candidate.skills == 1.0  # không yêu cầu kỹ năng -> phủ đầy


def test_rank_orders_and_breaks_ties_by_id():
    requirement = JobRequirement(semantic_base=1.0)
    profile = CandidateProfile()
    items = [
        ("cv_b", 0.5, profile),
        ("cv_a", 0.5, profile),
        ("cv_c", 0.9, profile),
    ]
    ranked = HybridRanker().rank(requirement, items)
    assert [c.doc_id for c in ranked] == ["cv_c", "cv_a", "cv_b"]
    assert all(isinstance(c, RankedCandidate) for c in ranked)


def test_rank_documents_uses_profiles_and_components():
    from cvjd.retrieval import RetrievalResult
    from cvjd.schema import Document, Section

    job_doc = Document(
        doc_id="jd_x", doc_type="jd", lang="vi", raw_text="python",
        sections={"skills": Section(name="skills", text="python")},
    )
    cv_doc = Document(
        doc_id="cv_x", doc_type="cv", lang="vi", raw_text="python 3 years",
        sections={
            "skills": Section(name="skills", text="python"),
            "experience": Section(name="experience", text="3 years"),
        },
    )
    results = [RetrievalResult("cv_x", 0.8, {"skills": 0.8})]
    ranked = HybridRanker().rank_documents(job_doc, results, {"cv_x": cv_doc})
    assert ranked[0].doc_id == "cv_x"
    assert ranked[0].section_components == {"skills": 0.8}
