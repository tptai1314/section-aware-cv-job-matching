"""Unit tests cho explanation + rerank (E17/E18/E19)."""

import pytest

from app import errors
from app.config import Settings
from app.explain import (
    explain_prompt,
    explain_with_llm,
    parse_explain,
    parse_rerank,
    rerank_candidate_prompt,
    rerank_with_llm,
)


def make_settings() -> Settings:
    return Settings(
        ollama_url="http://ollama:11434",
        ollama_model="qwen3:4b",
        ollama_timeout_s=5,
        max_upload_mb=1,
        max_pages=20,
        max_llm_chars=24000,
        ocr_enabled=False,
        ocr_langs="vie+eng",
        strict_evidence=False,
        allow_legacy_doc=False,
        embed_model="BAAI/bge-m3",
        embed_device="cpu",
        log_level="INFO",
    )


class FakeLLM:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def chat(self, system, user):
        self.calls += 1
        return self.payload


def test_parse_explain_ok():
    claims, summary, err = parse_explain({
        "summary": "Phù hợp cao",
        "claims": [
            {"statement": "Có React Native", "verdict": "hit", "evidence": "3 năm React Native", "source": "cv", "confidence": 0.9},
            {"statement": "Thiếu SQL", "verdict": "missing", "evidence": None, "source": None, "confidence": "0.8"},
            {"statement": "bad verdict", "verdict": "xyz", "evidence": "", "source": "cv", "confidence": None},
        ],
    })
    assert err is None
    assert summary == "Phù hợp cao"
    assert len(claims) == 3
    assert claims[0]["verdict"] == "hit"
    assert claims[1]["confidence"] == pytest.approx(0.8)
    assert claims[2]["verdict"] == "unknown"
    assert claims[2]["evidence"] is None
    assert claims[2]["source"] is None


def test_parse_explain_empty_and_non_object():
    assert parse_explain([]) == ([], None, "NOT_OBJECT")
    claims, summary, err = parse_explain({"claims": [{"statement": "  "}]})
    assert claims == [] and summary is None and err is None


def test_parse_explain_keeps_claims_without_sources():
    claims, _, _ = parse_explain(
        {"claims": [{"statement": "x", "verdict": "hit", "evidence": "không có ở đâu"}]}
    )
    assert claims[0]["verdict"] == "hit"


def test_parse_explain_verifies_evidence_grounding():
    job = {"required_skills": ["Docker"]}
    cv = {"skills": ["Node.js"]}
    claims, _, err = parse_explain(
        {
            "summary": "s",
            "claims": [
                {"statement": "Có Node", "verdict": "hit", "evidence": "Node.js", "source": "cv"},
                {"statement": "Bịa", "verdict": "hit", "evidence": "không có trong cv", "source": "cv"},
                {"statement": "Thiếu Docker", "verdict": "missing", "evidence": None, "source": None},
                {"statement": "Hit không source", "verdict": "hit", "evidence": "Docker"},
            ],
        },
        job,
        cv,
    )
    assert err is None
    assert claims[0]["verdict"] == "hit" and claims[0]["evidence"] == "Node.js"
    assert claims[1]["verdict"] == "unknown" and claims[1]["evidence"] is None
    assert claims[2]["verdict"] == "unknown"  # missing thiếu evidence -> hạ
    assert claims[3]["verdict"] == "hit"
    assert claims[3]["source"] == "job"  # suy ra nguồn từ evidence


def test_parse_rerank_filters_unknown_ids_and_clamps():
    out = parse_rerank({"c1", "c2"}, {"results": [
        {"id": "c1", "score": 120, "breakdown": {"a": 1}, "evidence": ["x"], "missing": ["y"], "reason": "r"},
        {"id": "ghost", "score": 80},            # id không kỳ vọng -> bỏ
        {"id": "c2", "score": None, "reason": "bad"},  # score None -> ready False
    ]})
    assert set(out) == {"c1", "c2"}
    assert out["c1"]["score"] == 100
    assert out["c1"]["ready"] is True
    assert out["c2"]["ready"] is False


def test_explain_with_llm_success():
    client = FakeLLM({"summary": "OK", "claims": [{"statement": "Có Node", "verdict": "hit", "evidence": "Node", "source": "cv"}]})
    res = explain_with_llm(client, make_settings(), {"skills": ["Node"]}, {"skills": ["Node"]}, None)
    assert res["ready"] is True
    assert res["fallback_reason"] is None
    assert res["claims"][0]["statement"] == "Có Node"


def test_explain_with_llm_fallback_on_error():
    class Boom(FakeLLM):
        def chat(self, system, user):
            raise errors.llm_unavailable()

    res = explain_with_llm(Boom(None), make_settings(), {}, {}, None)
    assert res["ready"] is False
    assert res["fallback_reason"] == "LLM_UNAVAILABLE"


def test_explain_with_llm_fallback_no_claims():
    client = FakeLLM({"summary": "x", "claims": []})
    res = explain_with_llm(client, make_settings(), {}, {}, None)
    assert res["ready"] is False
    assert res["fallback_reason"] == "NO_CLAIMS"


def test_rerank_with_llm_sorts_and_falls_back_per_candidate():
    class Seq(FakeLLM):
        def __init__(self):
            self.payload = None
            self.calls = 0

        def chat(self, system, user):
            self.calls += 1
            return {"results": [{"id": f"c{self.calls}", "score": {1: 70, 2: 90}[self.calls], "reason": "x", "evidence": [], "missing": []}]}

    client = Seq()
    out = rerank_with_llm(client, make_settings(), {"title": "Dev"},
                          [{"id": "c1"}, {"id": "c2"}])
    assert [o["id"] for o in out] == ["c2", "c1"]
    assert out[0]["score"] == 90


def test_rerank_with_llm_invalid_output_keeps_ready_false():
    class Bad(FakeLLM):
        def chat(self, system, user):
            return {"results": [{"id": "c9", "score": 50}]}  # id khác -> không khớp c1

    out = rerank_with_llm(Bad(None), make_settings(), {"title": "Dev"}, [{"id": "c1"}])
    assert out[0]["id"] == "c1"
    assert out[0]["ready"] is False
    assert out[0]["reason"] == "INVALID_OUTPUT"


def test_rerank_with_llm_llm_error_fallback():
    class Boom(FakeLLM):
        def chat(self, system, user):
            raise errors.llm_timeout()

    out = rerank_with_llm(Boom(None), make_settings(), {"title": "Dev"}, [{"id": "c1"}])
    assert out[0]["ready"] is False
    assert out[0]["reason"] == "LLM_TIMEOUT"


def test_prompts_include_content_and_note():
    s = make_settings()
    up = explain_prompt(s, {"title": "Dev"}, {"skills": ["Node"]}, None)
    assert "hit|missing|unknown" in up and "JOB" in up and "CV" in up
    rp = rerank_candidate_prompt(s, {"title": "Dev"}, {"id": "c7", "skills": ["node"]}, 0)
    assert "c7" in rp and "skills_required" in rp


def test_prompts_mark_untrusted_documents():
    from app import llm as llm_mod

    s = make_settings()
    up = explain_prompt(s, {"title": "Dev"}, {"skills": ["Node"]}, None)
    rp = rerank_candidate_prompt(s, {"title": "Dev"}, {"id": "c7"}, 0)
    assert llm_mod.UNTRUSTED_NOTE in up
    assert llm_mod.UNTRUSTED_NOTE in rp