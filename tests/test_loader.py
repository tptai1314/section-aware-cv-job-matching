"""Tests for the real-data loader and language detection."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cvjd.data.loader import (  # noqa: E402
    detect_lang,
    document_from_text,
    load_raw_dir,
    load_text_dir,
)
from cvjd.io import save_documents  # noqa: E402
from cvjd.schema import Document  # noqa: E402

_EN = "SUMMARY\nData scientist with 5 years of experience.\n\nSKILLS\npython, sql"
_VI = "TÓM TẮT\nKỹ sư dữ liệu với 5 năm kinh nghiệm.\n\nKỸ NĂNG\npython, sql"
_VI_HEAVY = (
    "Kinh nghiệm làm việc tại công ty phần mềm\n"
    "Phân tích dữ liệu và mô hình hóa, quản lý dự án\n"
    "Trình độ học vấn đại học chuyên ngành công nghệ thông tin"
)


def test_detect_lang():
    assert detect_lang(_EN) == "en"
    assert detect_lang(_VI_HEAVY) == "vi"
    assert detect_lang(_VI) in ("vi", "mixed")
    assert detect_lang("") == "en"


def test_document_from_text_splits_sections():
    doc = document_from_text("cv_real", "cv", _EN)
    assert doc.doc_id == "cv_real"
    assert doc.doc_type == "cv"
    assert doc.lang == "en"
    assert set(doc.sections) == {"summary", "skills"}
    assert doc.sections["skills"].text == "python, sql"


def test_load_text_dir(tmp_path):
    (tmp_path / "cv_a.txt").write_text(_EN, encoding="utf-8")
    (tmp_path / "cv_b.md").write_text(_VI_HEAVY, encoding="utf-8")
    (tmp_path / "ignore.pdf").write_bytes(b"%PDF")
    docs = load_text_dir(tmp_path, "cv")
    assert [d.doc_id for d in docs] == ["cv_a", "cv_b"]
    assert docs[1].lang == "vi"


def test_load_text_dir_duplicate_stem(tmp_path):
    (tmp_path / "cv_a.txt").write_text(_EN, encoding="utf-8")
    (tmp_path / "cv_a.md").write_text(_EN, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate doc_id"):
        load_text_dir(tmp_path, "cv")


def test_load_text_dir_empty(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_text_dir(tmp_path, "cv")


def test_load_raw_dir_layout_dir(tmp_path):
    raw = tmp_path / "raw"
    (raw / "cvs").mkdir(parents=True)
    (raw / "jds").mkdir()
    (raw / "cvs" / "cv_1.txt").write_text(_EN, encoding="utf-8")
    (raw / "jds" / "jd_1.txt").write_text(_VI_HEAVY, encoding="utf-8")
    cvs, jds = load_raw_dir(raw, layout="dir")
    assert [d.doc_id for d in cvs] == ["cv_1"]
    assert [d.doc_id for d in jds] == ["jd_1"]
    assert jds[0].doc_type == "jd"


def test_load_raw_dir_layout_jsonl(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    doc = Document(
        doc_id="cv_9", doc_type="cv", lang="en", raw_text=_EN,
        sections={},
    )
    save_documents(raw / "cvs.jsonl", [doc])
    save_documents(raw / "jds.jsonl", [doc.__class__(
        doc_id="jd_9", doc_type="jd", lang="en", raw_text=_EN, sections={},
    )])
    cvs, jds = load_raw_dir(raw, layout="jsonl")
    assert [d.doc_id for d in cvs] == ["cv_9"]
    assert [d.doc_id for d in jds] == ["jd_9"]


def test_load_raw_dir_missing_everything(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_raw_dir(tmp_path / "nope", layout="auto")


def test_load_raw_dir_bad_layout(tmp_path):
    with pytest.raises(ValueError, match="unknown layout"):
        load_raw_dir(tmp_path, layout="xml")
