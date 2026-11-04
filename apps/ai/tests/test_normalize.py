from app import normalize


def test_normalize_date_full():
    assert normalize.normalize_date("2022-03") == "2022-03"
    assert normalize.normalize_date("03/2022") == "2022-03"
    assert normalize.normalize_date("Mar 2022") == "2022-03"
    assert normalize.normalize_date("Tháng 3/2022") == "2022-03"
    assert normalize.normalize_date("2022") == "2022"


def test_normalize_date_invalid():
    assert normalize.normalize_date("không rõ") is None
    assert normalize.normalize_date("") is None
    assert normalize.normalize_date(None) is None
    assert normalize.normalize_date("13/2022") is None  # tháng 13 -> không suy diễn


def test_is_current_marker():
    assert normalize.is_current_marker("Present")
    assert normalize.is_current_marker("Hiện tại")
    assert normalize.is_current_marker("2022-03") is False


def test_dedupe_skills_case_insensitive():
    rows = [{"name": "React"}, {"name": "  react "}, {"name": "Vue"}]
    out = normalize.dedupe_skills(rows)
    assert [r["name"] for r in out] == ["React", "Vue"]


def test_dedupe_technologies():
    assert normalize.dedupe_technologies(["React", " React ", "React Native"]) == ["React", "React Native"]


def test_check_evidence_verbatim():
    skills = [{"name": "React", "confidence": 0.9, "evidence": "3 years building React apps"}]
    cv = "Worked 3 years building React apps and 1 in Vue."
    assert normalize.check_evidence(skills, cv, False) == []
    assert skills[0]["confidence"] == 0.9


def test_check_evidence_missing_lowers_confidence():
    skills = [{"name": "GraphQL", "confidence": 0.9, "evidence": "used GraphQL at Acme"}]
    cv = "Built frontend with Vue."
    assert normalize.check_evidence(skills, cv, False) == []
    assert skills[0]["confidence"] <= 0.3


def test_check_evidence_missing_strict_drops():
    skills = [{"name": "GraphQL", "confidence": 0.9, "evidence": "used GraphQL at Acme"}]
    assert normalize.check_evidence(skills, "Built frontend with Vue.", True) == ["GraphQL"]
    assert skills == []


def test_clamp_warns():
    warnings = []
    items = [{"n": i} for i in range(5)]
    out = normalize.clamp(items, 3, warnings, "skills")
    assert len(out) == 3
    assert warnings == ["LIMIT_SKILLS_TRIMMED"]