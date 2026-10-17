"""Tiếp nhận file (magic bytes, giới hạn) và trích xuất/làm sạch văn bản (spec mục 4–5)."""

from contextlib import contextmanager
import os
import re
import tempfile
import unicodedata

from . import errors
from .config import Settings

MAX_PDF_EXTRA_MB = 16  # trang un-compress có thể lớn hơn file thô


@contextmanager
def temp_file(raw: bytes):
    fd, path = tempfile.mkstemp(suffix=".upload")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(raw)
        yield path
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def sniff(kind: str, head: bytes) -> bool:
    if kind == "pdf":
        return head.startswith(b"%PDF-")
    if kind == "docx":
        return head.startswith(b"PK\x03\x04")  # zip container; entry word/document.xml check sau
    if kind == "doc":
        return head.startswith(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1")
    return False


def classify(head: bytes, allow_legacy_doc: bool) -> str:
    if sniff("pdf", head):
        return "pdf"
    if sniff("docx", head):
        return "docx"
    if allow_legacy_doc and sniff("doc", head):
        return "doc"
    raise errors.cv_file_invalid()


async def read_upload(upload, settings: Settings) -> tuple[bytes, str]:
    """Đọc file + xác định loại bằng magic bytes. Trả (raw, kind)."""
    if upload is None or upload.size == 0:
        raise errors.cv_file_required()
    limit = settings.max_upload_mb * 1024 * 1024
    raw = await upload.read(limit + 1)
    if len(raw) > limit:
        raise errors.cv_file_too_large()
    kind = classify(raw[:16], settings.allow_legacy_doc)
    if kind == "docx":
        # zip hợp lệ phải có word/document.xml
        import zipfile
        import io

        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                if "word/document.xml" not in zf.namelist():
                    raise errors.cv_file_invalid()
        except errors.ApiError:
            raise
        except Exception:
            raise errors.cv_file_invalid()
    return raw, kind


# ---- Làm sạch văn bản (spec 5.3) ----

_BULLET = re.compile(r"^\s*[•●▪‣☑✓]\s*")
_HEADER_FOOTER = re.compile(r"^\s*(page|trang|p\.?)\s*\d+\s*,?\s*$", re.IGNORECASE)
_WS = re.compile(r"[ \t\r\f\v]+")
_PAGE_BREAK = re.compile(r"\f")


def clean_text(text: str) -> str:
    if not text:
        return ""
    # 1. NFKC + bỏ control chars, chuẩn hoá newline, nbsp, zero-width
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\xad", "").replace("\u200b", "").replace("\u200c", "").replace("\u200d", "")
    text = text.replace("\xa0", " ")
    text = _PAGE_BREAK.sub("\n", text)
    text = "".join(ch for ch in text if ch == "\n" or ch == "\t" or ord(ch) >= 32)
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    lines = []
    for raw in text.split("\n"):
        line = _BULLET.sub("- ", raw)
        if _HEADER_FOOTER.search(line):
            continue
        line = _WS.sub(" ", line).strip()
        if line:
            lines.append(line.rstrip())
        else:
            lines.append("")

    # gộp nhiều dòng trống liên tiếp -> giữ 1
    out: list[str] = []
    blank = 0
    for line in lines:
        if line:
            out.append(line)
            blank = 0
        else:
            blank += 1
            if blank == 1:
                out.append("")
    return "\n".join(out).strip()


def merge_broken_lines(text: str) -> str:
    """Nối dòng bị ngắt: line trước chưa kết thúc (không có dấu câu cuối) và line sau
    bắt đầu bằng chữ thường/bullet -> ghép (spec 5.3)."""
    lines = [ln for ln in text.split("\n")]
    merged: list[str] = []
    for i, ln in enumerate(lines):
        if merged and ln and not _is_paragraph_start(ln) and _continues(merged[-1]):
            merged[-1] = merged[-1] + " " + ln.strip()
        else:
            merged.append(ln)
    return "\n".join(merged)


def _is_paragraph_start(line: str) -> bool:
    stripped = line.lstrip()
    if stripped.startswith("- "):
        return True
    # chữ thường ở đầu dòng = tiếp nối câu trước
    return not stripped or stripped[0].isupper() or stripped[0].isdigit()


def _continues(prev: str) -> bool:
    s = prev.rstrip()
    if not s:
        return False
    if s.endswith((".", "!", "?", ":", ";", ",", "—", "–", "-")):
        return False
    return True


# ---- Trích xuất theo loại file ----

def _extract_pdf(path: str, settings: Settings) -> tuple[str, list[str]]:
    from pypdf import PdfReader

    reader = PdfReader(path)
    if reader.is_encrypted:
        raise errors.cv_encrypted()
    pages = reader.pages
    if len(pages) > settings.max_pages:
        pages = pages[: settings.max_pages]
    parts = []
    for idx, page in enumerate(pages):
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            parts.append("")
    return "\n".join(f"\f{parts[i]}" for i in range(len(parts))), ["truncated"]


def _extract_docx(path: str) -> str:
    from docx import Document

    doc = Document(path)
    body_text: list[str] = []
    for child in doc.element.body.iterchildren():
        tag = child.tag.split("}")[-1]
        if tag == "p":
            from docx.text.paragraph import Paragraph
            from docx.oxml.ns import qn

            para = Paragraph(child, doc)
            body_text.append(para.text)
        elif tag == "tbl":
            from docx.table import Table

            for row in Table(child, doc).rows:
                cells = [c.text.strip() for c in row.cells]
                body_text.append(" | ".join(cells))
    return "\n".join(body_text)


def extract_text(path: str, kind: str, settings: Settings) -> tuple[str, int, bool, list[str]]:
    """Trả (text, pages, ocr_used, warnings). Trạng thái source do caller quyết định."""
    warnings: list[str] = []
    ocr_used = False
    truncated = False
    pages = 0

    if kind == "pdf":
        text, _ = "", []
        from pypdf import PdfReader

        reader = PdfReader(path)
        if reader.is_encrypted:
            raise errors.cv_encrypted()
        total = len(reader.pages)
        pages = min(total, settings.max_pages)
        truncated = total > settings.max_pages
        parts = []
        for idx, page in enumerate(reader.pages[:pages]):
            try:
                parts.append(page.extract_text() or "")
            except Exception:
                parts.append("")
        text = "\f".join(parts)

        per_page = [len(p) for p in parts]
        if _looks_scanned(per_page):
            if settings.ocr_enabled:
                ocr_text = _ocr_pages(path, pages, settings, warnings)
                if ocr_text.strip():
                    text = ocr_text
                    ocr_used = True
            else:
                warnings.append("OCR_LOW_QUALITY")
        if truncated:
            warnings.append("TRUNCATED")
        return text, pages, ocr_used, warnings

    if kind == "docx":
        text = _extract_docx(path)
        return text, 0, False, warnings

    # DOC cũ: chỉ hỗ trợ khi có LibreOffice headless
    if kind == "doc" and settings.allow_legacy_doc:
        return _convert_doc(path, warnings)

    raise errors.cv_file_invalid()


def _looks_scanned(chars_per_page: list[int]) -> bool:
    if not chars_per_page:
        return False
    non_empty = [c for c in chars_per_page if c > 0]
    if not non_empty:
        return True
    avg = sum(non_empty) / len(non_empty)
    return avg < 100 or sum(1 for c in chars_per_page if c == 0) > 0.6 * len(chars_per_page)


def _ocr_pages(path: str, pages: int, settings: Settings, warnings: list[str]) -> str:
    try:
        import pytesseract
        from pdf2image import convert_from_path
    except ImportError:
        warnings.append("OCR_LOW_QUALITY")
        return ""
    try:
        images = convert_from_path(path, dpi=300, first_page=1, last_page=pages)
        parts = [pytesseract.image_to_string(img, lang=settings.ocr_langs) for img in images]
        return "\f".join(parts)
    except Exception:
        warnings.append("OCR_LOW_QUALITY")
        return ""


def _convert_doc(path: str, warnings: list[str]) -> str:
    import subprocess

    if subprocess.run(["which", "libreoffice"], capture_output=True).returncode != 0:
        raise errors.cv_file_invalid("CV_FILE_INVALID")
    out = path + ".docx"
    try:
        subprocess.run(
            ["libreoffice", "--headless", "--convert-to", "docx", "--outdir", os.path.dirname(path), path],
            capture_output=True,
            timeout=60,
        )
        return _extract_docx(out)
    except Exception:
        raise errors.cv_file_invalid("CV_FILE_INVALID")
    finally:
        try:
            os.remove(out)
        except OSError:
            pass