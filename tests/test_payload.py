"""Tests cvjd.data.payload: payload đã trích xuất -> Document (sections + raw_text)."""

import pytest

from cvjd.data.payload import document_from_payload


def test_cv_payload_builds_sections():
    doc = document_from_payload(
        "cv_1",
        "cv",
        {
            "summary": "Backend dev",
            "skills": [{"name": "Python"}, {"name": "Postgres"}],
            "experience": [{"title": "Backend", "company": "Acme", "description": "APIs"}],
            "projects": [{"name": "Shop", "technologies": ["FastAPI"]}],
            "education": [{"degree": "BSc", "school": "HUST"}],
        },
    )
    assert doc.doc_type == "cv"
    assert set(doc.sections) == {"skills", "experience", "projects", "education", "summary"}
    assert doc.sections["skills"].text == "Python, Postgres"
    assert "Acme" in doc.sections["experience"].text
    assert "FastAPI" in doc.sections["projects"].text
    assert doc.raw_text


def test_job_payload_builds_sections():
    doc = document_from_payload(
        "jd_1",
        "jd",
        {
            "title": "Backend Engineer",
            "required_skills": [{"name": "Node"}, {"name": "Postgres"}],
            "optional_skills": [{"name": "Docker"}],
            "responsibilities": ["Build APIs", "Write tests"],
        },
    )
    assert doc.sections["skills"].text == "Node, Postgres"
    assert doc.sections["optional_skills"].text == "Docker"
    assert "Build APIs" in doc.sections["responsibilities"].text
    assert doc.sections["summary"].text == "Backend Engineer"


def test_payload_raw_text_fallback_when_empty():
    doc = document_from_payload("cv_x", "cv", {})
    assert doc.raw_text == "cv_x"
    assert doc.sections == {}


def test_payload_rejects_bad_type():
    with pytest.raises(ValueError):
        document_from_payload("x", "resume", {})
