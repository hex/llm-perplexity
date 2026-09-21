# ABOUTME: Shared pytest fixtures for the llm-perplexity test suite.
# ABOUTME: Isolates llm's user directory and keeps API keys out of recorded cassettes.
import os

import pytest


@pytest.fixture(autouse=True)
def perplexity_key(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_USER_PATH", str(tmp_path))
    monkeypatch.setenv(
        "LLM_PERPLEXITY_KEY", os.environ.get("LLM_PERPLEXITY_KEY", "pplx-test-key")
    )


@pytest.fixture(scope="module")
def vcr_config():
    return {"filter_headers": ["authorization"]}
