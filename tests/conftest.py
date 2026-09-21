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


DROPPED_RESPONSE_HEADERS = ("set-cookie", "x-request-id", "cf-ray")


def drop_tracking_headers(response):
    response["headers"] = {
        name: value
        for name, value in response["headers"].items()
        if name.lower() not in DROPPED_RESPONSE_HEADERS
        and not name.lower().startswith("x-ratelimit-")
    }
    return response


@pytest.fixture(scope="module")
def vcr_config():
    return {
        "filter_headers": ["authorization"],
        "match_on": ["method", "scheme", "host", "port", "path", "query", "body"],
        "decode_compressed_response": True,
        "before_record_response": drop_tracking_headers,
    }
