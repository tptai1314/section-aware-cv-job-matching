"""Tests cvjd.ranking.structured: payload `extract-cv`/`extract-job` -> ranking objects."""

from datetime import datetime, timezone

from cvjd.ranking.hybrid import hybrid_score, skills_score
from cvjd.ranking.structured import (
    budget_range,
    candidate_profile,
    job_requirement,
    skill_items,
    years_from_experience,
)


def test_skill_items_accepts_dict_and_str():
    items = skill_items(
        [
            {"name": "React", "proficiency_level": "9", "weight": 2},
            {"name": "", "proficiency_level": 3},
            "Node.js",
        ]
    )
    assert [i.name for i in items] == ["React", "Node.js"]
    assert items[0].level is None  # "9" (str) bị bỏ, giữ int hợp lệ
    assert items[0].weight == 2.0
    assert items[1].weight == 1.0


def test_years_from_experience_sums_and_current():
    entries = [
        {"start_date": "2020-01", "end_date": "2022-01"},
        {"start_date": "2022-06", "end_date": "Present", "is_current": True},
    ]
    years = years_from_experience(entries, now=datetime(2024, 6, 1, tzinfo=timezone.utc))
    assert years == 4.0  # 2 năm + 2 năm
    assert years_from_experience([]) is None


def test_budget_range_none_when_empty():
    assert budget_range(None) is None
    assert budget_range({"min": None, "max": None}) is None
    assert budget_range({"min": 10, "max": 20, "currency": "USD"}).max == 20.0


def test_job_requirement_reads_extract_job_fields():
    job = job_requirement(
        {
            "required_skills": [{"name": "React"}, {"name": "Node"}],
            "optional_skills": [{"name": "Docker"}],
            "min_years_experience": "3",
            "budget": {"min": 10, "max": 20, "currency": "USD"},
            "preferences": {"remote": "true"},
        }
    )
    assert [s.name for s in job.required_skills] == ["React", "Node"]
    assert job.min_years_experience == 3.0
    assert job.budget.max == 20.0
    assert job.preferences == {"remote": "true"}


def test_candidate_profile_computes_years_from_experience():
    cv = candidate_profile(
        {
            "skills": [{"name": "React"}],
            "experience": [{"start_date": "2021-01", "end_date": "2024-01"}],
        }
    )
    assert cv.years_experience == 3.0
    assert [s.name for s in cv.skills] == ["React"]


def test_structured_scores_end_to_end():
    job = job_requirement(
        {
            "required_skills": [{"name": "React"}, {"name": "Node"}],
            "min_years_experience": 3,
        }
    )
    strong = candidate_profile(
        {
            "skills": [{"name": "reactjs"}, {"name": "node.js"}],
            "years_experience": 5,
        }
    )
    weak = candidate_profile({"skills": [{"name": "PHP"}], "years_experience": 1})
    assert skills_score(strong, job) == 1.0
    assert skills_score(weak, job) == 0.0
    assert hybrid_score(job, strong)["overall"] > hybrid_score(job, weak)["overall"]
