"""Tests hybrid ranking (plan §7): S_hybrid weighted, components trong [0,1]."""

from cvjd.ranking.hybrid import (
    DEFAULT_WEIGHTS,
    BudgetRange,
    CandidateProfile,
    JobRequirement,
    SkillItem,
    budget_score,
    experience_score,
    hybrid_score,
    job_from_document,
    normalize_skill,
    preferences_score,
    profile_from_document,
    skills_score,
)
from cvjd.schema import Document, Section

EMPTY = CandidateProfile()
JOB = JobRequirement(required_skills=[SkillItem(name="React")], min_years_experience=3)


def test_weights_sum_to_one():
    assert abs(sum(DEFAULT_WEIGHTS.values()) - 1.0) < 1e-9


def test_normalize_skill_alias():
    assert normalize_skill("ReactJS") == "react"
    assert normalize_skill("  Node.JS ") == "node.js"
    assert normalize_skill("Javascript") == "js"


def test_skills_score_full_and_missing():
    job = JobRequirement(required_skills=[SkillItem(name="React"), SkillItem(name="Node.js")])
    full = CandidateProfile(skills=[SkillItem(name="React"), SkillItem(name="Node.js")])
    assert skills_score(full, job) == 1.0
    none = CandidateProfile(skills=[SkillItem(name="Go")])
    assert skills_score(none, job) == 0.0
    partial = CandidateProfile(skills=[SkillItem(name="React")])
    assert 0.0 < skills_score(partial, job) < 1.0


def test_skills_score_required_dominates_optional():
    job = JobRequirement(
        required_skills=[SkillItem(name="React")],
        optional_skills=[SkillItem(name="Vue")],
    )
    only_req = CandidateProfile(skills=[SkillItem(name="React")])
    only_opt = CandidateProfile(skills=[SkillItem(name="Vue")])
    assert skills_score(only_req, job) > skills_score(only_opt, job)


def test_experience_score():
    assert experience_score(CandidateProfile(years_experience=3), JOB) == 1.0
    assert experience_score(CandidateProfile(years_experience=1.5), JOB) == 0.5
    assert experience_score(EMPTY, JOB) == 0.0
    assert experience_score(EMPTY, JobRequirement(min_years_experience=None)) == 1.0


def test_budget_score():
    job = JobRequirement(budget=BudgetRange(min=100, max=200))
    inside = CandidateProfile(budget=BudgetRange(min=120, max=180))
    assert budget_score(inside, job) == 0.6
    no_cv = CandidateProfile()
    assert budget_score(no_cv, job) == 1.0  # không rõ ngân sách = trung lập
    far = CandidateProfile(budget=BudgetRange(min=1000, max=2000))
    assert budget_score(far, job) == 0.0


def test_preferences_score():
    job = JobRequirement(preferences={"remote": "yes", "timezone": "Asia/Ho_Chi_Minh"})
    good = CandidateProfile(preferences={"remote": "Yes", "timezone": "Asia/Ho_Chi_Minh"})
    assert preferences_score(good, job) == 1.0
    half = CandidateProfile(preferences={"remote": "yes"})
    assert preferences_score(half, job) == 0.5
    assert preferences_score(CandidateProfile(), job) == 0.0
    assert preferences_score(CandidateProfile(), JobRequirement()) == 1.0


def test_hybrid_score_higher_for_better_candidate():
    base = JobRequirement(
        required_skills=[SkillItem(name="React")],
        min_years_experience=3,
        budget=BudgetRange(min=100, max=200),
        semantic_base=0.5,
    )
    good = CandidateProfile(
        skills=[SkillItem(name="React")],
        years_experience=5,
        budget=BudgetRange(min=150, max=180),
    )
    poor = CandidateProfile(
        skills=[SkillItem(name="Go")],
        years_experience=0,
        budget=BudgetRange(min=1000, max=2000),
    )
    g = hybrid_score(base, good)["overall"]
    p = hybrid_score(base, poor)["overall"]
    assert g > p
    assert 0.0 <= g <= 1.0 and 0.0 <= p <= 1.0


def test_hybrid_score_defaults_component_weights():
    out = hybrid_score(JOB, EMPTY)
    assert set(out) == {"overall", "semantic", "skills", "experience", "budget", "preferences"}
    expected = (
        0.40 * 1.0  # semantic
        + 0.30 * skills_score(EMPTY, JOB)
        + 0.15 * 0.0
        + 0.10 * 1.0
        + 0.05 * 1.0
    )
    assert abs(out["overall"] - expected) < 1e-9


def test_semantic_base_clamped():
    out = hybrid_score(JobRequirement(semantic_base=1.7), EMPTY)
    assert out["semantic"] == 1.0
    out2 = hybrid_score(JobRequirement(semantic_base=-0.3), EMPTY)
    assert out2["semantic"] == 0.0


def test_profile_and_job_from_document_bridge():
    doc = Document(
        doc_id="c1",
        doc_type="cv",
        lang="vi",
        raw_text="React developer 5 nam",
        sections={
            "skills": Section(name="skills", text="React, Node.js"),
            "experience": Section(name="experience", text="5 nam lam React tai ABC"),
        },
    )
    profile = profile_from_document(doc)
    assert {"react", "node.js"} <= {s.name for s in profile.skills}
    assert profile.years_experience == 5.0

    jd = Document(
        doc_id="j1",
        doc_type="jd",
        lang="vi",
        raw_text="Can 2 nam React",
        sections={
            "skills": Section(name="skills", text="React"),
            "experience": Section(name="experience", text="toi thieu 2 nam kinh nghiem"),
        },
    )
    req = job_from_document(jd)
    assert req.required_skills[0].name == "react"
    assert req.min_years_experience == 2.0


def test_normalize_skill_aliases():
    assert normalize_skill("ci cd") == "ci/cd"
    assert normalize_skill("Docker Container") == "docker"
    assert normalize_skill("Tailwind CSS") == "tailwind css"
    assert normalize_skill("amazon web services") == "aws"


def test_skills_score_must_have_penalty():
    job = JobRequirement(
        required_skills=[SkillItem(name="React"), SkillItem(name="Node.js"), SkillItem(name="Postgres"), SkillItem(name="Docker")]
    )
    # Candidate matches 1 out of 4 (req = 0.25 < 0.5)
    cand = CandidateProfile(skills=[SkillItem(name="React")])
    score_normal = skills_score(cand, job, must_have_penalty=False)
    score_penalized = skills_score(cand, job, must_have_penalty=True)
    assert score_normal == 0.8 * 0.25 + 0.2 * 0.25  # 0.25
    assert score_penalized == score_normal * 0.25  # 0.0625
    assert score_penalized < score_normal


def test_hybrid_score_renormalize_missing():
    # Job only has skills and semantic, no budget/preferences/experience requirements
    job = JobRequirement(
        required_skills=[SkillItem(name="React")],
        semantic_base=0.8,
        min_years_experience=None,
        budget=None,
        preferences={},
    )
    cand = CandidateProfile(
        skills=[SkillItem(name="React")],
        years_experience=None,
        budget=None,
        preferences={},
    )
    # Default without renormalize: budget(1.0) and pref(1.0) inject arbitrary weight
    out_default = hybrid_score(job, cand, renormalize_missing=False)
    assert "overall" in out_default

    # With renormalize_missing=True: active weights are semantic (0.4) and skills (0.3)
    out_renorm = hybrid_score(job, cand, renormalize_missing=True)
    # Semantic=0.8, skills=1.0 -> (0.4*0.8 + 0.3*1.0) / (0.4 + 0.3) = 0.62 / 0.7 = 0.8857
    expected = (0.40 * 0.8 + 0.30 * 1.0) / (0.40 + 0.30)
    assert abs(out_renorm["overall"] - expected) < 1e-6


def test_hybrid_score_skill_gating():
    # Job requires React and Python. Candidate only knows Figma (0% skill coverage).
    job = JobRequirement(
        required_skills=[SkillItem(name="React"), SkillItem(name="Python")],
        semantic_base=0.70,
    )
    unrelated_cand = CandidateProfile(skills=[SkillItem(name="Figma")])

    # Without gating: semantic 0.70 + default fallbacks would produce artificially high match
    out_ungated = hybrid_score(job, unrelated_cand, must_have_penalty=False, renormalize_missing=True)
    assert out_ungated["skills"] == 0.0
    assert out_ungated["overall"] > 0.30  # ~0.40

    # With skill gating: score is gated by 0.15 + 0.85*(0) = 0.15
    out_gated = hybrid_score(job, unrelated_cand, must_have_penalty=True, renormalize_missing=True)
    assert out_gated["overall"] < 0.10  # Plummets down to < 10%
    assert abs(out_gated["overall"] - out_ungated["overall"] * 0.15) < 1e-6

    # When candidate has >= 50% required skills (e.g. 1 out of 2 = 50%), gate is 1.0 (no penalty)
    partial_cand = CandidateProfile(skills=[SkillItem(name="React")])
    out_partial_ungated = hybrid_score(job, partial_cand, must_have_penalty=False, renormalize_missing=True)
    out_partial_gated = hybrid_score(job, partial_cand, must_have_penalty=True, renormalize_missing=True)
    assert abs(out_partial_gated["overall"] - out_partial_ungated["overall"]) < 1e-6

    # When candidate has 100% required skills, gate is 1.0 (no penalty)
    perfect_cand = CandidateProfile(skills=[SkillItem(name="React"), SkillItem(name="Python")])
    out_perfect_ungated = hybrid_score(job, perfect_cand, must_have_penalty=False, renormalize_missing=True)
    out_perfect_gated = hybrid_score(job, perfect_cand, must_have_penalty=True, renormalize_missing=True)
    assert abs(out_perfect_gated["overall"] - out_perfect_ungated["overall"]) < 1e-6


def test_profile_from_document_section_evidence_weights():
    doc = Document(
        doc_id="c_ev",
        doc_type="cv",
        lang="en",
        raw_text="Summary: Familiar with Docker.\nSkills: React, Python\nExperience: Built APIs with React and AWS.",
        sections={
            "skills": Section(name="skills", text="React, Python"),
            "experience": Section(name="experience", text="Built APIs with React and AWS."),
        },
    )
    profile = profile_from_document(doc)
    weights = {s.name: s.weight for s in profile.skills}
    assert weights["react"] == 1.25  # corroborated in both skills & experience
    assert weights["python"] == 1.0
    assert weights["aws"] == 1.0
    assert weights["docker"] == 0.75  # only mentioned outside skills & experience

