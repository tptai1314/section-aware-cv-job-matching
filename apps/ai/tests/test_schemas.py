from app import schemas


def test_cv_extra_forbid():
    try:
        schemas.Skill(name="React", proficiency_level=9, extra_field=1)
        assert False, "phải lỗi extra=forbid"
    except Exception:
        pass


def test_skill_proficiency_out_of_range():
    import pydantic

    try:
        schemas.Skill(name="React", proficiency_level=11)
        assert False
    except pydantic.ValidationError:
        pass


def test_cv_response_limits():
    import pydantic

    try:
        schemas.CvResponse(
            skills=[schemas.Skill(name=f"s{i}") for i in range(schemas.LIMIT_SKILLS + 1)],
            meta=schemas.Meta(),
        )
        assert False
    except pydantic.ValidationError:
        pass


def test_meta_source_enum():
    import pydantic

    try:
        schemas.Meta(source="exe")
        assert False
    except pydantic.ValidationError:
        pass
    assert schemas.Meta(source="job_text").source == "job_text"
    assert schemas.Meta(source="pdf_ocr").source == "pdf_ocr"
    assert schemas.Meta(source="docx").source == "docx"


def test_embeddings_request_needs_texts():
    import pydantic

    try:
        schemas.EmbeddingsRequest(texts=[])
        assert False
    except pydantic.ValidationError:
        pass


def test_minimal_cv_response():
    resp = schemas.CvResponse(meta=schemas.Meta(source="pdf_text"))
    assert resp.skills == []
    assert resp.summary is None
    assert resp.meta.warnings == []