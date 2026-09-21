# ABOUTME: Tests for PerplexityOptions validation.
# ABOUTME: Covers rejected values and options the Agent API does not support.
import pytest

from llm_perplexity import PerplexityOptions


@pytest.mark.parametrize(
    "option_name,invalid_value",
    [
        ("temperature", 3.0),
        ("top_p", 2.0),
        ("search_recency_filter", "invalid_filter"),
        ("search_domain_filter", "not a domain"),
    ],
)
def test_invalid_option_values_are_rejected(option_name, invalid_value):
    with pytest.raises(ValueError):
        PerplexityOptions(**{option_name: invalid_value})


# Each value passes field validation on its own, so a failure proves the
# extra-field rejection.
UNSUPPORTED_OPTIONS = [
    ("top_k", 40),
    ("stream", False),
    ("presence_penalty", 0.5),
    ("frequency_penalty", 1.5),
    ("search_type", "fast"),
    ("search_mode", "academic"),
    ("disable_search", True),
    ("search_language_filter", "en"),
    ("return_images", True),
    ("return_related_questions", True),
    ("language_preference", "en"),
    ("stop", "END"),
    ("use_openrouter", True),
]


@pytest.mark.parametrize("option_name,unsupported_value", UNSUPPORTED_OPTIONS)
def test_unsupported_options_are_rejected(option_name, unsupported_value):
    import llm

    options_class = llm.get_model("sonar").Options
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        options_class(**{option_name: unsupported_value})


def test_options_logged_before_the_agent_api_still_load():
    import llm

    logged = {
        "temperature": 1.0,
        "stream": True,
        "return_related_questions": False,
        "include_citations": True,
        "use_openrouter": False,
    }
    options = llm.get_model("sonar").Options(**logged)
    assert options.temperature == 1.0
    assert options.include_citations is True
    assert not hasattr(options, "stream")


@pytest.mark.parametrize(
    "option_name,deliberate_value",
    [("stream", False), ("return_related_questions", True), ("use_openrouter", True)],
)
def test_unsupported_options_set_deliberately_are_still_rejected(option_name, deliberate_value):
    import llm

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        llm.get_model("sonar").Options(**{option_name: deliberate_value})


def test_search_context_size_accepts_documented_values():
    for size in ("low", "medium", "high"):
        assert PerplexityOptions(search_context_size=size).search_context_size == size


def test_search_context_size_rejects_other_values():
    with pytest.raises(ValueError):
        PerplexityOptions(search_context_size="huge")


def test_temperature_is_unset_by_default():
    assert PerplexityOptions().temperature is None


def test_temperature_zero_is_valid():
    assert PerplexityOptions(temperature=0).temperature == 0


def test_temperature_and_top_p_together_are_rejected():
    with pytest.raises(ValueError):
        PerplexityOptions(temperature=0.5, top_p=0.9)
