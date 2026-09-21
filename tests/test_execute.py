# ABOUTME: Cassette-backed tests for Perplexity.execute against the Agent API.
# ABOUTME: Covers non-streaming, streaming, citations, usage and API errors.
import json
import os

import llm
import pytest
import sqlite_utils
from llm.cli import load_conversation
from llm.migrations import migrate

from llm_perplexity import failure_message, integration_header, search_results


def test_search_results_come_from_the_search_results_output_item():
    data = {
        "output": [
            {
                "type": "search_results",
                "results": [{"id": 1, "title": "Example", "url": "https://example.com"}],
            },
            {"type": "message", "content": [{"type": "output_text", "text": "Hi"}]},
        ]
    }
    assert search_results(data) == [
        {"id": 1, "title": "Example", "url": "https://example.com"}
    ]


def test_search_results_is_empty_when_no_search_ran():
    assert search_results({"output": [{"type": "message", "content": []}]}) == []


def test_integration_header_names_the_plugin_and_version():
    name, _, version = integration_header().partition("/")
    assert name == "llm-perplexity"
    assert version


def test_failure_message_uses_the_error_the_api_reported():
    data = {"status": "failed", "error": {"code": "server_error", "message": "Model overloaded"}}
    assert failure_message(data) == "Model overloaded"


def test_failure_message_has_a_fallback_when_no_error_is_given():
    assert (
        failure_message({"status": "failed", "error": None})
        == "the response failed without an error message"
    )


@pytest.mark.parametrize("status", ["completed", "incomplete"])
def test_failure_message_is_none_for_responses_that_did_not_fail(status):
    assert failure_message({"status": status, "error": None}) is None


def test_format_citations_prints_the_url_alone_when_a_source_has_no_title():
    model = llm.get_model("sonar")
    formatted = model.format_citations([{"url": "https://example.com"}])
    assert formatted == "\n\n## Citations:\n[1] https://example.com\n"


@pytest.mark.vcr
def test_non_streaming_prompt():
    response = llm.get_model("sonar").prompt(
        "What is the capital of France? One sentence.", stream=False
    )
    text = response.text()
    assert "Paris" in text
    assert "## Citations:" in text
    assert "[1] " in text
    assert response.response_json["status"] == "completed"
    assert response.input_tokens > 0
    assert response.output_tokens > 0
    assert response._prompt_json == {
        "input": [{"role": "user", "content": "What is the capital of France? One sentence."}]
    }


@pytest.mark.vcr
def test_streaming_prompt_yields_several_chunks():
    response = llm.get_model("sonar").prompt(
        "What is the capital of France? One sentence."
    )
    chunks = list(response)
    assert len(chunks) > 2
    assert "Paris" in "".join(chunks)
    assert response.response_json["status"] == "completed"
    assert response.input_tokens > 0


@pytest.mark.vcr
def test_citations_can_be_left_out_of_the_text():
    response = llm.get_model("sonar").prompt(
        "What is the capital of France? One sentence.",
        stream=False,
        include_citations=False,
    )
    assert "## Citations:" not in response.text()
    assert search_results(response.response_json) != []


@pytest.mark.vcr
def test_follow_up_prompt_in_a_conversation():
    conversation = llm.get_model("sonar").conversation()
    conversation.prompt("What is the capital of France? One word.", stream=False).text()
    follow_up = conversation.prompt("And of Spain? One word.", stream=False)
    assert "Madrid" in follow_up.text()


@pytest.mark.vcr
def test_rejected_key_raises_model_error(monkeypatch):
    monkeypatch.setenv("LLM_PERPLEXITY_KEY", "pplx-invalid-key")
    with pytest.raises(llm.ModelError):
        llm.get_model("sonar").prompt("Hello", stream=False).text()


def test_a_conversation_logged_before_the_agent_api_still_loads_with_c(tmp_path):
    db_path = tmp_path / "logs.db"
    db = sqlite_utils.Database(str(db_path))
    migrate(db)
    db["conversations"].insert({"id": "conv1", "name": "capital", "model": "sonar"})
    db["responses"].insert(
        {
            "id": "resp1",
            "model": "sonar",
            "prompt": "Capital of France?",
            "system": None,
            "prompt_json": None,
            "options_json": json.dumps(
                {"temperature": 1.0, "stream": True, "use_openrouter": False}
            ),
            "response": "Paris.",
            "response_json": None,
            "conversation_id": "conv1",
            "duration_ms": 10,
            "datetime_utc": "2026-01-01T00:00:00",
            "input_tokens": 1,
            "output_tokens": 1,
            "token_details": None,
            "schema_id": None,
            "resolved_model": None,
            "reasoning": None,
        }
    )
    conversation = load_conversation("conv1", database=str(db_path))
    assert conversation.responses[0].prompt.options.temperature == 1.0


@pytest.mark.vcr
def test_a_key_passed_to_prompt_is_used(monkeypatch):
    monkeypatch.delenv("LLM_PERPLEXITY_KEY")
    response = llm.get_model("sonar").prompt(
        "What is the capital of France? One word.",
        stream=False,
        key=os.environ.get("PERPLEXITY_API_KEY", "pplx-test-key"),
    )
    assert "Paris" in response.text()


@pytest.mark.vcr
def test_search_and_generation_options_are_accepted_by_the_api():
    response = llm.get_model("sonar").prompt(
        "What is new in the llm CLI? One sentence.",
        stream=False,
        search_context_size="low",
        search_domain_filter="simonwillison.net",
        search_recency_filter="year",
        reasoning_effort="low",
        max_tokens=200,
    )
    response.text()
    assert response.response_json["status"] == "completed"
    sources = search_results(response.response_json)
    assert sources
    assert all("simonwillison.net" in source["url"] for source in sources)
