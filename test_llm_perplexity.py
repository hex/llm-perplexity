import os
import pytest
import llm
import tempfile
from pathlib import Path
from PIL import Image, ImageDraw
import dotenv

# Load environment variables from .env file
dotenv.load_dotenv()

# Skip tests if the API key isn't set
requires_api_key = pytest.mark.skipif(
    not os.environ.get("LLM_PERPLEXITY_KEY"),
    reason="LLM_PERPLEXITY_KEY environment variable not set",
)

@pytest.fixture
def temp_image():
    """Create a temporary test image with text for image-based tests."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        img = Image.new('RGB', (300, 200), color='white')
        draw = ImageDraw.Draw(img)
        draw.text((50, 80), "Hello Perplexity!", fill="black")
        img.save(tmp.name)
        tmp_path = tmp.name

    yield tmp_path

    os.unlink(tmp_path)

# Test standard models with parameterization
@requires_api_key
@pytest.mark.parametrize(
    "model_id,prompt",
    [
        ("sonar-pro", "Fun facts about AI"),
        ("sonar", "Fun facts about walruses"),
    ],
)
def test_standard_models(model_id, prompt):
    """Test standard models listed in the README."""
    model = llm.get_model(model_id)
    response = model.prompt(prompt, stream=False)
    assert response is not None
    assert len(response.text()) > 0

# Test other models with parameterization
@requires_api_key
@pytest.mark.parametrize(
    "model_id,prompt",
    [
        ("sonar-deep-research", "Complex research question"),
        ("sonar-reasoning-pro", "Problem solving task"),
    ],
)
def test_other_models(model_id, prompt):
    """Test other models mentioned in the README."""
    model = llm.get_model(model_id)
    response = model.prompt(prompt, stream=False)
    assert response is not None
    assert len(response.text()) > 0

# Test advanced options with parameterization
@requires_api_key
@pytest.mark.parametrize(
    "options,prompt",
    [
        ({"temperature": 0.7}, "Generate creative ideas"),
        ({"top_p": 0.9}, "Generate varied responses"),
        ({"top_k": 40}, "Generate focused content"),
        ({"max_tokens": 100}, "Summarize this article"),
        ({"return_related_questions": True}, "How does quantum computing work?"),
    ],
)
def test_advanced_options(options, prompt):
    """Test advanced options mentioned in the README."""
    model = llm.get_model("sonar-pro")
    response = model.prompt(prompt, stream=False, **options)
    assert response is not None
    assert len(response.text()) > 0

    # For max_tokens, check that response isn't too long
    if "max_tokens" in options:
        assert len(response.text()) < 2000  # Rough check that max_tokens is working

@requires_api_key
def test_image_analysis(temp_image):
    """Test analyzing an image."""
    model = llm.get_model("sonar-pro")
    prompt = llm.Prompt(
        "What can you tell me about this image?",
        model=model,
        options={"image_path": temp_image}
    )
    response = model.execute(prompt)

    assert response is not None
    assert len(response.text()) > 0

    text = response.text().lower()
    assert any(word in text for word in ["text", "image", "white", "hello", "perplexity"])

@requires_api_key
def test_streaming_response():
    """Test streaming responses."""
    model = llm.get_model("sonar-pro")
    response = model.prompt("Tell me about streaming data", stream=True)
    chunks = list(response)

    assert len(chunks) > 0
    assert "".join(chunks)

@requires_api_key
def test_pro_search_streaming():
    """Test Pro Search by enabling search_type with streaming."""
    model = llm.get_model("sonar-pro")
    response = model.prompt(
        "Summarize one recent AI research trend.",
        stream=True,
        search_type="pro",
    )
    chunks = list(response)
    assert len(chunks) > 0
    assert "".join(chunks)

# Test error handling for invalid options
@pytest.mark.parametrize(
    "option_name,invalid_value,expected_exception",
    [
        ("temperature", 3.0, ValueError),
        ("top_p", 2.0, ValueError),
        ("top_k", 3000, ValueError),
        ("search_recency_filter", "invalid_filter", ValueError),
    ],
)
def test_invalid_options(option_name, invalid_value, expected_exception):
    """Test error handling for invalid option values."""
    model = llm.get_model("sonar-pro")
    with pytest.raises(expected_exception):
        prompt = llm.Prompt(
            "This should fail",
            model=model,
            options={option_name: invalid_value}
        )
        model.execute(prompt)

@requires_api_key
def test_citations_handling():
    """Test that citations are properly handled."""
    model = llm.get_model("sonar-pro")
    response = model.prompt(
        "What are the latest developments in generative AI?",
        max_tokens=500,
        stream=False
    )
    assert response is not None
    text = response.text()
    assert len(text) > 0

@requires_api_key
def test_citations_excluding():
    """include_citations False should omit citations section and citation marks."""
    model = llm.get_model("sonar-pro")
    response = model.prompt(
        "What are the latest developments in generative AI?",
        max_tokens=500,
        stream=False,
        include_citations=False
    )
    text = response.text()
    assert text and len(text) > 0
    assert "## Citations:" not in text
    assert "[1]" not in text

# Only run OpenRouter test if the key is set
@pytest.mark.skipif(
    not os.environ.get("LLM_OPENROUTER_KEY"),
    reason="LLM_OPENROUTER_KEY environment variable not set",
)
def test_openrouter_access():
    """Test accessing models through OpenRouter."""
    model = llm.get_model("sonar")
    prompt = llm.Prompt(
        "Fun facts about pelicans",
        model=model,
        options={"use_openrouter": True}
    )
    response = model.execute(prompt)

    assert response is not None
    assert len(response.text()) > 0
