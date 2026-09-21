# ABOUTME: Cassette-backed tests for Perplexity.execute against the Agent API.
# ABOUTME: Covers non-streaming, streaming, citations, usage and API errors.
import llm
import pytest

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
        == "Perplexity reported that the response failed"
    )


@pytest.mark.parametrize("status", ["completed", "incomplete"])
def test_failure_message_is_none_for_responses_that_did_not_fail(status):
    assert failure_message({"status": status, "error": None}) is None


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
