"""Tests ChromaDB storage: IDs, upsert/replace, delete, metadata, query.

Lưu ý: chroma 1.5 EphemeralClient() dùng chung store trong-tiến-trình và
`reset()` bị tắt. Nên fixture client là session-scoped và các test dùng
document_id riêng + assertion theo từng tài liệu (không dựa vào global count).
"""

import pytest

from pathlib import Path

from cvjd.models.embedding import MockEmbedder
from cvjd.storage import chroma
from cvjd.schema import Section

REPO_SRC = str(Path(__file__).resolve().parents[1] / "src")

EMBED_MODEL = "mock"


@pytest.fixture(scope="session")
def client():
    return chroma.build_client(None)


@pytest.fixture(scope="session")
def embedder():
    return MockEmbedder(dim=32)


def test_collection_names_and_creation(client):
    names = chroma.ensure_collections(client)
    assert names == {"cv": chroma.CV_COLLECTION, "jd": chroma.JOB_COLLECTION}
    assert chroma.CV_COLLECTION == "frevia_cvs"
    assert chroma.JOB_COLLECTION == "frevia_jobs"
    assert (client.get_collection(chroma.CV_COLLECTION).metadata or {})["hnsw:space"] == "cosine"


def test_collection_name_rejects_unknown():
    with pytest.raises(ValueError):
        chroma.collection_name("exe")


def test_vector_id_roundtrip():
    vid = chroma.vector_id("cv", "123", 1, "skills")
    assert vid == "cv:123:v1:skills"
    assert chroma.parse_vector_id(vid) == ("cv", "123", 1, "skills")
    assert chroma.parse_vector_id("jd:9:v42:_full") == ("jd", "9", 42, "_full")
    with pytest.raises(ValueError):
        chroma.parse_vector_id("bad")


def test_section_hash_stable_and_distinct():
    assert chroma.section_hash("a b") == chroma.section_hash("a b")
    assert chroma.section_hash("a b") != chroma.section_hash("a c")


def test_section_vectors_includes_full():
    sections = {
        "skills": Section(name="skills", text=" Node.js, TypeScript "),
        "preamble": Section(name="preamble", text=""),
    }
    out = chroma.section_vectors(sections, raw_text="Full CV text.")
    assert out == {"skills": "Node.js, TypeScript", chroma.FULL_SECTION: "Full CV text."}


def test_upsert_creates_section_and_full_vectors(client, embedder):
    chroma.ensure_collections(client)
    sections = {"skills": "React", "experience": "Acme", chroma.FULL_SECTION: "Full CV."}
    stats = chroma.upsert_document(
        client, "cv", "u-101", sections, embedder.encode, EMBED_MODEL, version=1
    )
    assert stats["upserted"] == 3
    rows = chroma.get_document(client, "cv", "u-101", include=["metadatas"])
    assert {r["id"] for r in rows} == {
        "cv:u-101:v1:skills",
        "cv:u-101:v1:experience",
        "cv:u-101:v1:_full",
    }
    md = next(r for r in rows if r["id"].endswith(":skills"))["metadata"]
    assert md["document_id"] == "u-101"
    assert md["document_type"] == "cv"
    assert md["document_version"] == 1
    assert md["section"] == "skills"
    assert md["status"] == "READY"
    assert md["embedding_model"] == EMBED_MODEL
    assert md["content_hash"] == chroma.section_hash("React")


def test_upsert_same_version_replaces_sections(client, embedder):
    chroma.ensure_collections(client)
    first = {"skills": "React", chroma.FULL_SECTION: "Full CV."}
    chroma.upsert_document(client, "cv", "u-5", first, embedder.encode, EMBED_MODEL, version=1)
    second = {"skills": "Vue", "education": "HCMUT", chroma.FULL_SECTION: "Full CV v2."}
    stats = chroma.upsert_document(client, "cv", "u-5", second, embedder.encode, EMBED_MODEL, version=1)
    rows = chroma.get_document(client, "cv", "u-5", include=["metadatas"])
    ids = {r["id"] for r in rows}
    assert ids == {"cv:u-5:v1:skills", "cv:u-5:v1:education", "cv:u-5:v1:_full"}
    assert stats["deleted_leftover"] == 0
    rows = chroma.get_document(client, "cv", "u-5", include=["metadatas"])
    assert not any(r["id"].endswith(":experience") for r in rows)


def test_version_bump_replaces_old_version(client, embedder):
    chroma.ensure_collections(client)
    sections = {"skills": "React", chroma.FULL_SECTION: "Full."}
    chroma.upsert_document(client, "jd", "u-7", sections, embedder.encode, EMBED_MODEL, version=1)
    stats = chroma.upsert_document(client, "jd", "u-7", sections, embedder.encode, EMBED_MODEL, version=2)
    assert stats["deleted_stale"] == 2
    rows = chroma.get_document(client, "jd", "u-7", include=["metadatas"])
    ids = {r["id"] for r in rows}
    assert ids == {"jd:u-7:v2:skills", "jd:u-7:v2:_full"}
    assert all(r["id"].startswith("jd:u-7:v2:") for r in rows)


def test_delete_document(client, embedder):
    chroma.ensure_collections(client)
    sections = {"skills": "Go", chroma.FULL_SECTION: "Full."}
    chroma.upsert_document(client, "cv", "u-42", sections, embedder.encode, EMBED_MODEL, version=1)
    chroma.upsert_document(client, "cv", "u-43", sections, embedder.encode, EMBED_MODEL, version=1)
    assert chroma.delete_document(client, "cv", "u-42") == 2
    assert chroma.get_document(client, "cv", "u-42") == []
    assert len(chroma.get_document(client, "cv", "u-43")) == 2


def test_query_ranks_related_higher(client, embedder):
    chroma.ensure_collections(client)
    jobs = [
        ("w-1", {"skills": "React Node.js", chroma.FULL_SECTION: "Cần React developer."}),
        ("w-2", {"skills": "Marketing SEO", chroma.FULL_SECTION: "Tuyển digital marketing."}),
    ]
    for doc_id, sections in jobs:
        chroma.upsert_document(client, "jd", doc_id, sections, embedder.encode, EMBED_MODEL, version=1)

    query_embedding = embedder.encode(["React developer Node.js"])
    hits = chroma.query(client, "jd", query_embedding, n_results=50, status="READY")
    rank = {
        h["metadata"]["document_id"]: i
        for i, h in enumerate(hits)
        if h["metadata"]["document_id"] in ("w-1", "w-2")
    }
    assert rank["w-1"] < rank["w-2"]
    assert hits[0]["metadata"]["status"] == "READY"


def test_query_respects_where_status(client, embedder):
    chroma.ensure_collections(client)
    chroma.upsert_document(
        client, "jd", "w-a", {"skills": "React", chroma.FULL_SECTION: "A."},
        embedder.encode, EMBED_MODEL, version=1, status="READY",
    )
    chroma.upsert_document(
        client, "jd", "w-b", {"skills": "React", chroma.FULL_SECTION: "B."},
        embedder.encode, EMBED_MODEL, version=1, status="PROCESSING",
    )
    query_embedding = embedder.encode(["React"])
    hits = chroma.query(client, "jd", query_embedding, n_results=50, status="READY")
    ids = {h["metadata"]["document_id"] for h in hits}
    assert "w-a" in ids
    assert "w-b" not in ids


def test_persistent_store_reopen_roundtrip(tmp_path):
    """Mở lại từ "process khác": parent ghi + close, subprocess mở lại và phải
    thấy collections + vector. Chroma 1.5 đôi khi catalog chưa "dính" ngay sau
    write (Windows), nên subprocess thử lại vài lần qua build_client (warm-up)."""
    import subprocess
    import sys

    path = str(tmp_path / "chroma")
    client = chroma.build_client(path)
    chroma.ensure_collections(client)
    chroma.upsert_document(
        client, "jd", "p-1", {"skills": "Node.js", chroma.FULL_SECTION: "Full."},
        MockEmbedder(dim=32).encode, EMBED_MODEL, version=1,
    )
    client.close()

    code = (
        "import sys, time\n"
        "sys.path.insert(0, %r)\n"
        "from cvjd.storage import chroma\n"
        "import chromadb\n"
        "path = sys.argv[1]\n"
        "last = None\n"
        "for _ in range(4):\n"
        "    try:\n"
        "        c = chroma.build_client(path)\n"
        "        n = chroma.count_vectors(c, 'jd')\n"
        "        assert chroma.get_document(c, 'jd', 'p-1')[0]['id'] == 'jd:p-1:v1:skills'\n"
        "        print(n, flush=True)\n"
        "        sys.exit(0)\n"
        "    except chromadb.errors.NotFoundError as e:\n"
        "        last = e\n"
        "        time.sleep(1.0)\n"
        "print('still missing', last)\n"
        "sys.exit(3)\n"
    ) % str(REPO_SRC)
    proc = subprocess.run(
        [sys.executable, "-c", code, path],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout.strip() == "2"