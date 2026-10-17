"""Metadata filter (owner/visibility) ở tầng retrieval — plan §5.4/§7.4."""

from cvjd.models.embedding import MockEmbedder
from cvjd.retrieval.retriever import ChromaRetriever
from cvjd.schema import Document, Section
from cvjd.storage import chroma


def _cv(doc_id: str, text: str) -> Document:
    return Document(
        doc_id=doc_id,
        doc_type="cv",
        lang="vi",
        raw_text=text,
        sections={"skills": Section("skills", text)},
    )


def _retriever() -> ChromaRetriever:
    client = chroma.build_client(None)
    chroma.ensure_collections(client)
    return ChromaRetriever(client, MockEmbedder(dim=32), embed_model="mock")


def test_access_where_shapes():
    assert chroma.access_where() == {"visibility": {"$in": ["public"]}}
    assert chroma.access_where("u1") == {
        "$or": [{"visibility": {"$in": ["public"]}}, {"owner_id": {"$eq": "u1"}}]
    }
    assert chroma.access_where("u1", ()) == {"owner_id": {"$eq": "u1"}}
    assert chroma.access_where(visibilities=()) is None


def test_pool_filters_by_visibility():
    retriever = _retriever()
    retriever.index_corpus(
        [_cv("cv_public", "python postgres"), _cv("cv_private", "python fastapi")],
        meta={
            "cv_public": {"visibility": "public"},
            "cv_private": {"visibility": "private", "owner_id": "u2"},
        },
    )

    # người khác: thấy public, không thấy private (store Ephemeral dùng chung nên
    # các tài liệu public của test khác cũng có thể xuất hiện -> chỉ assert tập con)
    where_guest = chroma.access_where("u1", ("public",))
    pool = retriever.pool_by_text("python", "cv", pool_size=10, where=where_guest)
    guest_ids = {c["doc_id"] for c in pool}
    assert "cv_public" in guest_ids
    assert "cv_private" not in guest_ids

    # chủ sở hữu: thấy thêm tài liệu private của mình
    where_owner = chroma.access_where("u2", ("public",))
    pool_owner = retriever.pool_by_text("python", "cv", pool_size=10, where=where_owner)
    assert {"cv_public", "cv_private"} <= {c["doc_id"] for c in pool_owner}


def test_default_metadata_is_public():
    retriever = _retriever()
    retriever.index_corpus([_cv("cv_a", "go rust")])
    rows = chroma.get_document(retriever.client, "cv", "cv_a", include=["metadatas"])
    md = rows[0]["metadata"]
    assert md["visibility"] == "public"
    assert md["owner_id"] == ""
