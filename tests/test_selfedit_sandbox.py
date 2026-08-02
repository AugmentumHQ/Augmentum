"""Verify-time secret scrub (W11) — the verify subprocess must not inherit the
app's credentials, because it EXECUTES the candidate's code."""

from __future__ import annotations

from augmentum.selfedit.sandbox import is_secret_name, scrubbed_env


def test_drops_secret_shaped_names():
    for n in ("AUGMENTUM_OPENAI_API_KEY", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN",
              "DB_PASSWORD", "MY_SECRET", "AWS_CREDENTIAL", "X_APIKEY",
              "SESSION_ID", "AUTH_COOKIE"):
        assert is_secret_name(n), n


def test_keeps_the_things_a_subprocess_needs():
    for n in ("PATH", "HOME", "PYTHONPATH", "LANG", "TMPDIR", "VIRTUAL_ENV"):
        assert not is_secret_name(n), n
    # a non-secret app config var is kept
    assert not is_secret_name("AUGMENTUM_CLASSIFIER_BASE_URL")


def test_scrubbed_env_removes_secrets_keeps_rest():
    base = {
        "PATH": "/usr/bin", "HOME": "/root", "PYTHONPATH": "/app",
        "AUGMENTUM_OPENAI_API_KEY": "sk-secret", "HF_TOKEN": "hf-secret",
        "AUGMENTUM_MODE": "prod",  # non-secret config kept
    }
    out = scrubbed_env(base=base)
    assert "AUGMENTUM_OPENAI_API_KEY" not in out and "HF_TOKEN" not in out
    assert out["PATH"] == "/usr/bin" and out["PYTHONPATH"] == "/app"
    assert out["AUGMENTUM_MODE"] == "prod"


def test_extra_drop():
    base = {"PATH": "/x", "KEEP_ME": "1", "ALSO": "2"}
    out = scrubbed_env(base=base, extra_drop=("ALSO",))
    assert "ALSO" not in out and out["KEEP_ME"] == "1"


def test_default_reads_real_env_without_crashing():
    # smoke: over the real os.environ it returns a dict and never includes an
    # obviously-secret var if one is present
    out = scrubbed_env()
    assert isinstance(out, dict)
    assert not any(is_secret_name(k) for k in out)
