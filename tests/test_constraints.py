"""Tests hard constraints (plan §7.4): status/visibility/deleted/ownership."""

from cvjd.ranking.constraints import (
    AccessContext,
    CandidateRecord,
    apply_hard_constraints,
    is_allowed,
    public_candidates,
)

PUB = "public"
PRIV = "private"


def rec(doc_id="c1", **kw):
    fields = {"doc_id": doc_id, "doc_type": "cv"}
    fields.update(kw)
    return CandidateRecord(**fields)


def test_deleted_always_excluded():
    r = rec(deleted_at="2026-01-01")
    assert is_allowed(r, AccessContext()) is False
    assert is_allowed(r, AccessContext(is_admin=True)) is False


def test_non_ready_excluded_when_required():
    assert is_allowed(rec(status="PROCESSING"), AccessContext()) is False
    assert is_allowed(rec(status="PROCESSING"), AccessContext(), require_ready=False) is True


def test_private_only_owner_or_admin():
    r = rec(visibility=PRIV, owner_id="u1")
    assert is_allowed(r, AccessContext(user_id="u1")) is True
    assert is_allowed(r, AccessContext()) is False
    assert is_allowed(r, AccessContext(is_admin=True)) is True
    # private CV không owner -> không bao giờ public
    assert is_allowed(rec(visibility=PRIV, owner_id=None), AccessContext()) is False


def test_unlisted_role_gate():
    r = rec(visibility="unlisted", owner_id="u1")
    assert is_allowed(r, AccessContext(user_id="u1")) is True
    assert is_allowed(r, AccessContext(roles=frozenset({"recruiter"})), allowed_roles=["recruiter"]) is True
    assert is_allowed(r, AccessContext(roles=frozenset({"viewer"}))) is False


def test_public_anyone():
    assert is_allowed(rec(visibility=PUB), AccessContext()) is True


def test_apply_preserves_order_and_drops_invalid():
    records = [
        rec("a", visibility=PUB),
        rec("b", visibility=PRIV, owner_id="u1"),
        rec("c", doc_type="exe"),
        rec("d", status="PROCESSING"),
        rec("e", deleted_at="x"),
    ]
    kept = apply_hard_constraints(records, AccessContext(user_id="u1"))
    assert [r.doc_id for r in kept] == ["a", "b"]


def test_public_candidates():
    records = [
        rec("a", visibility=PUB),
        rec("b", visibility=PRIV, owner_id="u1"),
        rec("c", status="READY", visibility=PUB, deleted_at="x"),
    ]
    assert [r.doc_id for r in public_candidates(records, AccessContext())] == ["a"]