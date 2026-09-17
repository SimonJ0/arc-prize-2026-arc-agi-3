"""
Unit tests for the Cryptographic Iron Rule submission gate.
"""

import pytest

from src.submit.submission_gate import SubmissionAuthorizationGate


def test_build_submission_hash_output(tmp_path):
    gate = SubmissionAuthorizationGate()
    prov = gate.build_submission()
    assert "notebook_sha256" in prov
    assert len(prov["notebook_sha256"]) == 64
    assert prov["notebook_path"].endswith("submission.ipynb")


def test_request_approval_token_structure(tmp_path):
    gate = SubmissionAuthorizationGate()
    token = gate.request_approval(exp_id="TEST-001", notes="Unit test approval")
    assert "||" in token
    payload, sig = token.split("||")
    assert "TEST-001" in payload
    assert len(sig) == 64


def test_submit_fails_closed_on_tampered_token():
    gate = SubmissionAuthorizationGate()
    token = gate.request_approval(exp_id="TEST-FAIL-001")
    payload, sig = token.split("||")
    # Tamper with payload
    tampered = f"{payload}_TAMPERED||{sig}"
    with pytest.raises(SystemExit) as exc:
        gate.submit_with_approval(tampered)
    assert exc.value.code == 1


def test_submit_fails_closed_on_expired_token():
    gate = SubmissionAuthorizationGate()
    # Expired timestamp: 2020-01-01
    payload = "TEST-EXP|hash1|commit|cfg|eval|2020-01-01T00:00:00+00:00"
    import hashlib
    import hmac

    from src.submit.submission_gate import SECRET_SALT

    sig = hmac.new(SECRET_SALT, payload.encode("utf-8"), hashlib.sha256).hexdigest()
    expired_token = f"{payload}||{sig}"

    with pytest.raises(SystemExit) as exc:
        gate.submit_with_approval(expired_token)
    assert exc.value.code == 1
