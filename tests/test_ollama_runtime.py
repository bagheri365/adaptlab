from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace
from urllib import error as urlerror

from adaptlab.evaluation.runtime import capture_ollama_model_digest, capture_ollama_version


def test_capture_ollama_version_parses_local_cli_output(monkeypatch) -> None:
    def fake_run(*args, **kwargs):
        return SimpleNamespace(stdout="ollama version is 0.32.6\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert capture_ollama_version() == "0.32.6"


def test_capture_ollama_version_returns_none_when_runtime_missing(monkeypatch) -> None:
    def fake_run(*args, **kwargs):
        raise OSError("ollama not installed")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert capture_ollama_version() is None


def test_capture_ollama_model_digest_returns_matching_digest(monkeypatch) -> None:
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(
                {
                    "models": [
                        {"name": "other", "digest": "1" * 64},
                        {"name": "qwen3:8b", "digest": "2" * 64},
                    ]
                }
            ).encode("utf-8")

    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: FakeResponse())
    assert capture_ollama_model_digest("http://localhost:11434", "qwen3:8b") == "2" * 64


def test_capture_ollama_model_digest_returns_none_when_metadata_unavailable(monkeypatch) -> None:
    def fake_urlopen(*args, **kwargs):
        raise urlerror.URLError("offline")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    assert capture_ollama_model_digest("http://localhost:11434", "qwen3:8b") is None
