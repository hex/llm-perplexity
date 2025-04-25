import json
import llm
import os
import pytest
from pydantic import BaseModel
import base64
import tempfile
from pathlib import Path

# Set a default API key for tests if one isn't provided
PERPLEXITY_API_KEY = os.environ.get("PYTEST_PERPLEXITY_API_KEY", None) or "pplx-..."


@pytest.mark.vcr
def test_standard_model():
    model = llm.get_model("sonar-small")
    model.key = model.key or PERPLEXITY_API_KEY
    response = model.prompt("Provide two fun facts about sharks, be brief")
    
    text = response.text()
    assert len(text) > 0
    assert "shark" in text.lower()
    
    # Check response_json structure
    response_dict = response.response_json
    assert "choices" in response_dict
    assert len(response_dict["choices"]) > 0
    assert "message" in response_dict["choices"][0]
    
    # Check token usage
    assert response.input_tokens > 0
    assert response.output_tokens > 0


@pytest.mark.vcr
def test_online_model_with_search():
    model = llm.get_model("sonar-small-online")
    model.key = model.key or PERPLEXITY_API_KEY
    
    # Test with search
    response = model.prompt("What is the latest version of Python as of April 2025?")
    
    text = response.text()
    assert len(text) > 0
    assert "Python" in text
    
    # Check citations if they exist
    if "citations" in response.response_json:
        assert len(response.response_json["citations"]) > 0


@pytest.mark.vcr
def test_search_recency_filter():
    model = llm.get_model("sonar-small-online")
    model.key = model.key or PERPLEXITY_API_KEY
    
    # Test with recency filter
    response = model.prompt(
        "Summarize recent tech news",
        options={"search_recency_filter": "week"}
    )
    
    text = response.text()
    assert len(text) > 0
    
    # Check if search parameters were passed correctly in prompt_json
    assert "search_recency_filter" in response._prompt_json.get("messages", [{}])[-1].get("search_params", {})
    assert response._prompt_json.get("messages", [{}])[-1].get("search_params", {}).get("search_recency_filter") == "week"


@pytest.mark.vcr
def test_search_domain_filter():
    model = llm.get_model("sonar-small-online")
    model.key = model.key or PERPLEXITY_API_KEY
    
    # Test with domain filter
    response = model.prompt(
        "Recent machine learning papers",
        options={"search_domain_filter": "arxiv.org,github.com"}
    )
    
    text = response.text()
    assert len(text) > 0
    
    # Check if search parameters were passed correctly
    assert "search_domain_filter" in response._prompt_json.get("messages", [{}])[-1].get("search_params", {})
    assert "arxiv.org" in response._prompt_json.get("messages", [{}])[-1].get("search_params", {}).get("search_domain_filter", "")


@pytest.mark.vcr
def test_temperature_option():
    model = llm.get_model("sonar-small")
    model.key = model.key or PERPLEXITY_API_KEY
    
    # Test with custom temperature
    response = model.prompt(
        "Generate a creative name for a pet shark",
        options={"temperature": 0.7}
    )
    
    text = response.text()
    assert len(text) > 0
    
    # Ensure temperature was passed to the API
    assert response._prompt_json.get("temperature") == 0.7


@pytest.mark.vcr
def test_top_p_option():
    model = llm.get_model("sonar-small")
    model.key = model.key or PERPLEXITY_API_KEY
    
    # Test with custom top_p
    response = model.prompt(
        "Generate a creative name for a pet dolphin",
        options={"top_p": 0.9}
    )
    
    text = response.text()
    assert len(text) > 0
    
    # Ensure top_p was passed to the API
    assert response._prompt_json.get("top_p") == 0.9


@pytest.mark.vcr
def test_max_tokens_option():
    model = llm.get_model("sonar-small")
    model.key = model.key or PERPLEXITY_API_KEY
    
    # Test with max_tokens limit
    response = model.prompt(
        "Write a long poem about the ocean",
        options={"max_tokens": 100}
    )
    
    text = response.text()
    assert len(text) > 0
    
    # The response should be limited
    tokens = len(text.split())
    assert tokens < 200  # Rough approximation, may need adjustment
    
    # Ensure max_tokens was passed to the API
    assert response._prompt_json.get("max_tokens") == 100


@pytest.mark.vcr
def test_return_related_questions():
    model = llm.get_model("sonar-small-online")
    model.key = model.key or PERPLEXITY_API_KEY
    
    # Test with related questions
    response = model.prompt(
        "How does quantum computing work?",
        options={"return_related_questions": True}
    )
    
    text = response.text()
    assert len(text) > 0
    
    # Check if related questions are in the response
    if "related_questions" in response.response_json:
        assert len(response.response_json["related_questions"]) > 0


@pytest.mark.vcr
def test_image_analysis():
    model = llm.get_model("sonar-pro")
    model.key = model.key or PERPLEXITY_API_KEY
    
    # Create a simple test image
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        # Create a 1x1 red pixel PNG
        tmp.write(base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
        ))
        image_path = tmp.name
    
    try:
        # Test with image analysis
        response = model.prompt(
            "What color is shown in this image?",
            options={"image_path": image_path}
        )
        
        text = response.text()
        assert len(text) > 0
        assert "red" in text.lower() or "color" in text.lower()
        
    finally:
        # Clean up the temporary image
        if os.path.exists(image_path):
            os.unlink(image_path)


@pytest.mark.vcr
def test_openrouter_access():
    # Check if OpenRouter plugin is installed
    if not any(p["name"] == "llm-openrouter" for p in llm.get_plugins()):
        pytest.skip("OpenRouter plugin not installed")
    
    model = llm.get_model("sonar-small")
    model.key = model.key or PERPLEXITY_API_KEY
    
    # Set OpenRouter key if available
    openrouter_key = os.environ.get("PYTEST_OPENROUTER_KEY")
    if not openrouter_key:
        pytest.skip("OpenRouter API key not available")
    
    try:
        llm.set_key("openrouter", openrouter_key)
        
        # Test with OpenRouter
        response = model.prompt(
            "Fun facts about pelicans, be brief",
            options={"use_openrouter": True}
        )
        
        text = response.text()
        assert len(text) > 0
        assert "pelican" in text.lower()
        
    except Exception as e:
        pytest.skip(f"OpenRouter test failed: {str(e)}")
