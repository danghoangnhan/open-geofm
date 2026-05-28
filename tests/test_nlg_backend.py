"""NLG backend selector tests."""

from __future__ import annotations

import pytest

from open_geofm.nlg import backend, verify


def test_get_rewriter_unknown_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPEN_GEOFM_REWRITER", "bogus")
    with pytest.raises(ValueError, match="Unknown OPEN_GEOFM_REWRITER"):
        backend.get_rewriter()


def test_get_rewriter_openai_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPEN_GEOFM_REWRITER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    rw = backend.get_rewriter()
    # `rewrite` should fail because no key is set (lazy check, not at construction).
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        rw._ensure_client()  # type: ignore[attr-defined]


def test_verify_accepts_matching_number() -> None:
    res = verify.verify("Therefore AC = 5", "5")
    assert res.accepted
    assert res.extracted_answer == "5"


def test_verify_rejects_mismatching_number() -> None:
    res = verify.verify("Therefore AC = 4", "5")
    assert not res.accepted


def test_verify_accepts_sqrt_expressions() -> None:
    # FGPS may return symbolic; the NL solution typically gives a decimal.
    res = verify.verify("AC = 0.8660", "sqrt(3)/2")
    assert res.accepted


def test_verify_rejects_when_no_number_in_text() -> None:
    res = verify.verify("the proof concludes here", "5")
    assert not res.accepted
    assert "no number found" in res.reason
