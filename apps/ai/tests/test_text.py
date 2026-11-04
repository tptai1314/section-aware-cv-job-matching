from app import text


def test_classify_magic_bytes():
    assert text.classify(b"%PDF-1.7", False) == "pdf"
    assert text.classify(b"PK\x03\x04rest", False) == "docx"
    assert text.classify(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1", True) == "doc"
    try:
        text.classify(b"PK\x05\x06notzip", False)
        assert False, "phải raise"
    except Exception as exc:
        assert exc.detail == "CV_FILE_INVALID"


def test_sniff():
    assert text.sniff("pdf", b"%PDF-")
    assert text.sniff("docx", b"PK\x03\x04")
    assert text.sniff("doc", b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1")
    assert not text.sniff("pdf", b"nope")


def test_clean_text_bullets_and_whitespace():
    raw = "•  React\n\xa0\n•  Node.js\x0c\n Page 3"
    cleaned = text.clean_text(raw)
    assert "React" in cleaned
    assert "- Node.js" in cleaned
    assert "Page 3" not in cleaned
    assert "\xa0" not in cleaned


def test_clean_text_merges_blank_lines():
    raw = "a\n\n\n\nb"
    cleaned = text.clean_text(raw)
    assert cleaned.count("\n\n") == 1


def test_merge_broken_lines():
    raw = "Joined a startup and\nworked on the design system and\n- lead the team\nFinished."
    out = text.merge_broken_lines(raw)
    assert "startup and worked on the design system and" in out
    assert "- lead the team" in out