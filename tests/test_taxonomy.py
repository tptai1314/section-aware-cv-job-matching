"""Unit tests for the new Skill Taxonomy & Asymmetric Skill Extractor."""

import pytest
from cvjd.data.taxonomy import get_taxonomy
from cvjd.ranking.hybrid import (
    CandidateProfile,
    JobRequirement,
    SkillItem,
    _coverage,
    _match,
    normalize_skill,
    skills_score,
)
from cvjd.schema import Document, Section


def test_taxonomy_loaded():
    tax = get_taxonomy()
    assert len(tax.entries) >= 50
    assert "react" in tax.skills_by_id
    assert "python" in tax.skills_by_id


def test_taxonomy_disambiguation_c_and_variants():
    tax = get_taxonomy()

    # C++ and C#
    t1 = "Tôi có kinh nghiệm phát triển với C++ và C# trên nền tảng .NET."
    skills1 = tax.extract_skills_from_text(t1)
    assert "C++" in skills1
    assert "C#" in skills1
    assert ".NET" in skills1
    # Không được nhận nhầm thành C đơn lẻ
    assert "C" not in skills1

    # Ngôn ngữ C thực sự
    t2 = "Lập trình nhúng với ngôn ngữ C và Linux driver."
    skills2 = tax.extract_skills_from_text(t2)
    assert "C" in skills2
    assert "Linux" in skills2


def test_taxonomy_disambiguation_java_and_javascript():
    tax = get_taxonomy()
    t = "Thành thạo JavaScript, TypeScript, React trên frontend."
    skills = tax.extract_skills_from_text(t)
    assert "JavaScript" in skills
    assert "TypeScript" in skills
    assert "React" in skills
    assert "Java" not in skills


def test_taxonomy_disambiguation_go_and_r():
    tax = get_taxonomy()
    t1 = "Senior Go developer with microservices background."
    s1 = tax.extract_skills_from_text(t1)
    assert "Go" in s1
    assert "Microservices" in s1

    # Động từ go thường không được nhận
    t2 = "We want candidates who can go the extra mile and do great things."
    s2 = tax.extract_skills_from_text(t2)
    assert "Go" not in s2

    # R programming
    t3 = "Data analysis using R language and Python pandas."
    s3 = tax.extract_skills_from_text(t3)
    assert "R" in s3
    assert "Python" in s3
    assert "Pandas" in s3


def test_taxonomy_english_aliases():
    tax = get_taxonomy()
    t = "Relational database administration, automated software testing, and cybersecurity."
    skills = tax.extract_skills_from_text(t)
    assert "Database" in skills
    assert "Software Testing" in skills
    assert "Cybersecurity" in skills


def test_jd_must_vs_nice_classification():
    tax = get_taxonomy()
    jd_text = """
    Job Description: Backend Software Engineer.
    
    Required Qualifications:
    - At least 2 years of experience with Python and Django.
    - Proficient in PostgreSQL and Redis.
    - Solid understanding of Docker and CI/CD.
    
    Nice to have (Preferred):
    - Experience with React or Next.js.
    - Familiarity with AWS and Kubernetes.
    """
    must, nice = tax.classify_jd_skills(jd_text)
    assert "Python" in must
    assert "Django" in must
    assert "PostgreSQL" in must
    assert "Docker" in must

    assert "React" in nice
    assert "Next.js" in nice
    assert "AWS" in nice
    assert "Kubernetes" in nice

    # Không bị giao nhau
    assert set(must).isdisjoint(set(nice))


def test_partial_credit_for_related_skills():
    # Job requires React
    job = JobRequirement(required_skills=[SkillItem(name="React")])

    # 1. Exact match gets 1.0
    cand_exact = CandidateProfile(skills=[SkillItem(name="React")])
    assert skills_score(cand_exact, job) == 1.0

    # 2. Next.js is framework_of React (weight 0.60)
    cand_related = CandidateProfile(skills=[SkillItem(name="Next.js")])
    score_rel = skills_score(cand_related, job)
    assert score_rel == 0.60

    # 3. Completely unrelated skill (e.g. Flutter) -> gets 0.0
    cand_unrel = CandidateProfile(skills=[SkillItem(name="Flutter")])
    assert skills_score(cand_unrel, job) == 0.0


def test_three_tier_matching_user_example():
    """Kiểm thử trực tiếp ví dụ thực tế:
    CV có: React, Next.js, TypeScript
    JD yêu cầu: React, Redux, JavaScript
    - React khớp tuyệt đối (Lớp 1): 1.0
    - TypeScript là is_superset_of của JavaScript (Lớp 2): 0.80
    - Next.js KHÔNG tương đương Redux (Lớp 2): 0.0 (không cho điểm sai lệch)
    """
    job = JobRequirement(
        required_skills=[
            SkillItem(name="React"),
            SkillItem(name="Redux"),
            SkillItem(name="JavaScript"),
        ]
    )
    cv = CandidateProfile(
        skills=[
            SkillItem(name="React"),
            SkillItem(name="Next.js"),
            SkillItem(name="TypeScript"),
        ]
    )
    # Tổng điểm: (1.0 + 0.0 + 0.80) / 3 = 1.80 / 3 = 0.60
    score = skills_score(cv, job)
    assert abs(score - 0.60) < 1e-6


def test_taxonomy_frozen_version_and_checksum():
    tax = get_taxonomy()
    assert tax.version == "1.0.0-frozen"
    assert len(tax.sha256) == 64  # Chuỗi hash sha256 chuẩn 64 ký tự hex


def test_asymmetric_coverage_with_nice_to_have():
    job = JobRequirement(
        required_skills=[SkillItem(name="Python")],
        optional_skills=[SkillItem(name="Docker")],
    )
    # Cand 1 has only must-have (Python) -> 0.8 * 1.0 + 0.2 * 0.0 = 0.8
    c1 = CandidateProfile(skills=[SkillItem(name="Python")])
    s1 = skills_score(c1, job)

    # Cand 2 has both must-have and nice-to-have -> 0.8 * 1.0 + 0.2 * 1.0 = 1.0
    c2 = CandidateProfile(skills=[SkillItem(name="Python"), SkillItem(name="Docker")])
    s2 = skills_score(c2, job)

    # Cand 3 has only nice-to-have -> 0.8 * 0.0 + 0.2 * 1.0 = 0.2
    c3 = CandidateProfile(skills=[SkillItem(name="Docker")])
    s3 = skills_score(c3, job)

    assert s2 > s1 > s3
    assert s2 == 1.0
    assert s1 == 0.8
    assert s3 == 0.2


def test_classify_jd_skills_single_paragraph_does_not_leak_nice_mode():
    """Inline 'is a plus' or 'preferred' must not flip subsequent sentences to nice-to-have."""
    tax = get_taxonomy()
    jd_text = (
        "Requirements: 8+ yrs as a Business Analyst. "
        "Scrum Master experience is a plus. "
        "Strong knowledge of AWS, Azure, and PostgreSQL is required."
    )
    must, nice = tax.classify_jd_skills(jd_text)
    assert "Business Analysis" in must
    assert "AWS" in must
    assert "Azure" in must
    assert "PostgreSQL" in must
    assert "Scrum Master" in nice


def test_classify_jd_skills_negation_guard():
    """Explicitly negated requirement clauses must not be extracted as required skills."""
    tax = get_taxonomy()
    jd_text = (
        "Required: Python and Docker.\n"
        "No prior Kubernetes experience required.\n"
        "Nice to have: Redis."
    )
    must, nice = tax.classify_jd_skills(jd_text)
    assert "Python" in must
    assert "Docker" in must
    assert "Kubernetes" not in must
    assert "Redis" in nice


def test_two_hop_relational_inference_and_non_leakage():
    """2-hop transitive chains (NestJS -> TypeScript -> JavaScript) give decayed credit,
    while unrelated pairs (Next.js != Redux, Flutter != React) stay 0.0."""
    tax = get_taxonomy()
    w_2hop = tax.get_relation_weight("NestJS", "JavaScript", max_hops=2)
    assert 0.20 < w_2hop < 0.40
    assert tax.get_relation_weight("Next.js", "Redux") == 0.0
    assert tax.get_relation_weight("Flutter", "React") == 0.0


def test_score_skills_coverage_boundary_renormalization_and_nice_l2():
    """Missing nice-to-have renormalizes to must_cov (no 0.85 ceiling), and nice uses L2."""
    tax = get_taxonomy()
    # Only must-have present, 100% covered -> must be 1.0 (not 0.85)
    assert tax.score_skills_coverage(["Python"], [], ["Python"]) == 1.0
    # Only nice-to-have present, L2 match (Next.js -> React = 0.60) -> 0.60
    assert abs(tax.score_skills_coverage([], ["React"], ["Next.js"]) - 0.60) < 1e-6
    # Empty JD skills -> neutral 1.0
    assert tax.score_skills_coverage([], [], ["Python"]) == 1.0


def test_layer3_semantic_skill_bridge_and_idf():
    """Optional Layer 3 semantic embeddings provide bounded partial credit for OOV/unlinked skills."""
    import numpy as np

    tax = get_taxonomy()

    def mock_encode(texts):
        vecs = []
        for t in texts:
            tl = t.lower()
            if "figma" in tl or "adobe xd" in tl or "ui design" in tl or "prototyping" in tl:
                vecs.append([0.9, 0.43589, 0.0])
            else:
                vecs.append([0.0, 0.0, 1.0])
        return np.asarray(vecs, dtype=np.float32)

    try:
        tax.attach_skill_embeddings(mock_encode, include_category=False)
        # Figma and Adobe XD have empty `relations` in JSON, so L2=0.0, L3 > 0.0
        assert tax.get_relation_weight("Figma", "Adobe XD") == 0.0
        w_l3 = tax.get_effective_relation_weight("Figma", "Adobe XD", use_layer3=True)
        assert 0.15 <= w_l3 <= 0.45
        # OOV skill phrase maps via semantic vector
        w_oov = tax.get_effective_relation_weight("UI Prototyping Tool", "Figma", use_layer3=True)
        assert w_oov > 0.0

        idf = tax.compute_skill_idf(["Python and Git", "Python and Git", "Ansible and Git"])
        assert idf["Ansible"] > idf["Git"]
    finally:
        tax._skill_embeddings = None
        tax._encode_fn = None
        tax._oov_emb_cache.clear()

