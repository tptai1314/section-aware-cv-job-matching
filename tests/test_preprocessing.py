"""Tests for the rule-based section splitter."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cvjd.data.preprocessing import (  # noqa: E402
    normalize_heading,
    resplit_document,
    sections_to_dict,
    split_sections,
)
from cvjd.schema import Document, Section  # noqa: E402


@pytest.mark.parametrize("heading,expected", [
    ("SUMMARY", "summary"),
    ("Profile", "summary"),
    ("Tóm tắt bản thân", "summary"),
    ("TECHNICAL SKILLS", "skills"),
    ("Kỹ năng", "skills"),
    ("WORK EXPERIENCE", "experience"),
    ("Kinh nghiệm làm việc", "experience"),
    ("EDUCATION", "education"),
    ("Trình độ học vấn", "education"),
    ("Projects", None),
    ("", None),
])
def test_normalize_heading(heading, expected):
    assert normalize_heading(heading) == expected


def test_normalize_heading_strips_colon():
    assert normalize_heading("SKILLS:") == "skills"


def test_split_sections_basic():
    text = "SUMMARY\nEngineer with 3 years.\n\nSKILLS\npython, sql"
    sections = split_sections(text)
    assert [s.name for s in sections] == ["summary", "skills"]
    assert sections[0].text == "Engineer with 3 years."
    assert sections[1].heading_raw == "SKILLS"
    assert sections[1].order == 1


def test_split_sections_preamble_before_first_heading():
    text = "Some intro line.\n\nEXPERIENCE\n- 2020-2023: Dev"
    sections = split_sections(text)
    assert sections[0].name == "preamble"
    assert sections[0].text == "Some intro line."
    assert sections[1].name == "experience"


def test_split_sections_unknown_heading_uses_slug():
    text = "PROJECTS\nBuilt a chatbot."
    sections = split_sections(text)
    assert sections[0].name == "projects"
    assert sections[0].heading_raw == "PROJECTS"


def test_split_sections_empty_text():
    assert split_sections("") == []
    assert split_sections("   \n  \n") == []


def test_sections_to_dict_suffixes_duplicates():
    dup = [Section("skills", "a", "SKILLS", 0), Section("skills", "b", "SKILLS", 1)]
    out = sections_to_dict(dup)
    assert set(out) == {"skills", "skills_2"}
    assert out["skills_2"].text == "b"


def test_resplit_document_preserves_identity_fields():
    text = "SUMMARY\nHi.\n\nSKILLS\npython"
    from cvjd.schema import Document

    doc = Document(doc_id="cv_x", doc_type="cv", lang="en", raw_text=text)
    rebuilt = resplit_document(doc)
    assert rebuilt.doc_id == doc.doc_id
    assert rebuilt.raw_text == doc.raw_text
    assert set(rebuilt.sections) == {"summary", "skills"}


@pytest.mark.parametrize("raw_text,expected", [
    (
        "PROFILE\nData Scientist with 3 years of experience.\n\n"
        "TECHNICAL SKILLS\npython, sql\n\n"
        "WORK EXPERIENCE\n- 2020-2023: Dev at FPT\n\n"
        "ACADEMIC BACKGROUND\n- Bachelor degree, Computer Science",
        [
            ("summary", "Data Scientist with 3 years of experience.", "PROFILE"),
            ("skills", "python, sql", "TECHNICAL SKILLS"),
            ("experience", "- 2020-2023: Dev at FPT", "WORK EXPERIENCE"),
            ("education", "- Bachelor degree, Computer Science", "ACADEMIC BACKGROUND"),
        ],
    ),
    (
        "TÓM TẮT\nKỹ sư dữ liệu với 3 năm kinh nghiệm.\n\n"
        "KỸ NĂNG\npython, sql\n\n"
        "KINH NGHIỆM LÀM VIỆC\n- 2019-2021: Dev tại Viettel\n\n"
        "TRÌNH ĐỘ HỌC VẤN\n- Cử nhân khoa học máy tính",
        [
            ("summary", "Kỹ sư dữ liệu với 3 năm kinh nghiệm.", "TÓM TẮT"),
            ("skills", "python, sql", "KỸ NĂNG"),
            ("experience", "- 2019-2021: Dev tại Viettel", "KINH NGHIỆM LÀM VIỆC"),
            ("education", "- Cử nhân khoa học máy tính", "TRÌNH ĐỘ HỌC VẤN"),
        ],
    ),
])
def test_resplit_document_round_trip(raw_text, expected):
    sections = {
        name: Section(name, text, heading, order)
        for order, (name, text, heading) in enumerate(expected)
    }
    doc = Document(doc_id="cv_x", doc_type="cv", lang="en",
                   raw_text=raw_text, sections=sections)
    rebuilt = resplit_document(doc)
    assert rebuilt.doc_id == doc.doc_id
    assert rebuilt.raw_text == doc.raw_text
    got = [
        (s.name, s.text, s.heading_raw)
        for s in sorted(rebuilt.sections.values(), key=lambda s: s.order)
    ]
    assert got == expected
