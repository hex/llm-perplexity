# ABOUTME: Tests for PerplexityOptions validation.
# ABOUTME: Covers rejected values and options that the Agent API no longer supports.
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
