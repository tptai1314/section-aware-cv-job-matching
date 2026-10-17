from app.llm import OllamaClient


def test_parse_json_plain():
    assert OllamaClient._parse_json('{"a": 1}') == {"a": 1}


def test_parse_json_code_fence():
    assert OllamaClient._parse_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_parse_json_with_prose():
    done = OllamaClient._parse_json('Đây là kết quả:\n{"skills": [], "summary": null}')
    assert done == {"skills": [], "summary": None}


def test_parse_json_invalid():
    assert OllamaClient._parse_json("just text, no json") is None
    assert OllamaClient._parse_json("") is None


def test_ping_unreachable():
    from app.config import Settings

    settings = Settings(
        ollama_url="http://127.0.0.1:59999",
        ollama_model="qwen3:4b",
        ollama_timeout_s=2,
        max_upload_mb=10,
        max_pages=20,
        max_llm_chars=24000,
        ocr_enabled=False,
        ocr_langs="vie+eng",
        strict_evidence=True,
        allow_legacy_doc=False,
        embed_model="BAAI/bge-m3",
        embed_device="cpu",
        log_level="INFO",
    )
    client = OllamaClient(settings)
    assert client.ping() is False
    client.close()