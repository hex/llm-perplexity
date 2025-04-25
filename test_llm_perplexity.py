import os
import pytest
import llm
import tempfile
from pathlib import Path
from PIL import Image, ImageDraw
import dotenv
import traceback
import sys

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
        # Create a simple test image with text
        img = Image.new('RGB', (300, 200), color='white')
        draw = ImageDraw.Draw(img)
        
        # Add some text to the image
        draw.text((50, 80), "Hello Perplexity!", fill="black")
        img.save(tmp.name)
        tmp_path = tmp.name
    
    yield tmp_path
    
    # Clean up the temp file
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
    try:
        print(f"\nTesting model: {model_id}")
        model = llm.get_model(model_id)
        print(f"Model instance: {model}")
        response = model.prompt(prompt, stream=False)
        assert response is not None
        assert len(response.text()) > 0
        print(f"Test passed for {model_id}")
    except Exception as e:
        print(f"Error testing {model_id}: {str(e)}")
        traceback.print_exc(file=sys.stdout)
        raise

# Test online models and search filters
@requires_api_key
@pytest.mark.parametrize(
    "model_id,prompt,options",
    [
        # Basic online model
        ("sonar-pro-online", "Latest AI research in 2025", {}),
    ],
)
def test_online_models_and_filters(model_id, prompt, options):
    """Test online models with various search filters."""
    try:
        print(f"\nTesting online model: {model_id} with options: {options}")
        model = llm.get_model(model_id)
        print(f"Model instance: {model}")
        response = model.prompt(prompt, stream=False, **options)
        assert response is not None
        assert len(response.text()) > 0
        print(f"Test passed for {model_id} with options: {options}")
    except Exception as e:
        print(f"Error testing {model_id} with options {options}: {str(e)}")
        traceback.print_exc(file=sys.stdout)
        raise

# Test other models with parameterization
@requires_api_key
@pytest.mark.parametrize(
    "model_id,prompt",
    [
        # Test one of each category to minimize API costs
        ("sonar-deep-research", "Complex research question"),
        ("sonar-reasoning-pro", "Problem solving task"),
        ("sonar-reasoning", "Logical reasoning"),
        ("r1-1776", "Fun facts about seals"),
    ],
)
def test_other_models(model_id, prompt):
    """Test other models mentioned in the README."""
    try:
        print(f"\nTesting other model: {model_id}")
        model = llm.get_model(model_id)
        print(f"Model instance: {model}")
        # Add a timeout to prevent hanging
        response = model.prompt(prompt, stream=False)
        assert response is not None
        assert len(response.text()) > 0
        print(f"Test passed for {model_id}")
    except Exception as e:
        print(f"Error testing {model_id}: {str(e)}")
        traceback.print_exc(file=sys.stdout)
        raise

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
    try:
        print(f"\nTesting advanced options: {options}")
        model = llm.get_model("sonar-pro")
        print(f"Model instance: {model}")
        response = model.prompt(prompt, stream=False, **options)
        assert response is not None
        assert len(response.text()) > 0
        print(f"Test passed with options: {options}")
        
        # For max_tokens, check that response isn't too long
        if "max_tokens" in options:
            assert len(response.text()) < 2000  # Rough check that max_tokens is working
    except Exception as e:
        print(f"Error testing with options {options}: {str(e)}")
        traceback.print_exc(file=sys.stdout)
        raise

@requires_api_key
def test_image_analysis(temp_image):
    """Test analyzing an image."""
    try:
        print(f"\nTesting image analysis")
        model = llm.get_model("sonar-pro")
        print(f"Model instance: {model}")
        
        # Create a model-specific prompt
        prompt = llm.Prompt(
            "What can you tell me about this image?",
            model=model,
            options={"image_path": temp_image}
        )
        response = model.execute(prompt)
        
        assert response is not None
        assert len(response.text()) > 0
        
        # The model should mention text or something about the image
        text = response.text().lower()
        assert any(word in text for word in ["text", "image", "white", "hello", "perplexity"])
        print(f"Image analysis test passed")
    except Exception as e:
        print(f"Error testing image analysis: {str(e)}")
        traceback.print_exc(file=sys.stdout)
        raise

@requires_api_key
def test_streaming_response():
    """Test streaming responses."""
    try:
        print(f"\nTesting streaming response")
        model = llm.get_model("sonar-pro")
        print(f"Model instance: {model}")
        response = model.prompt("Tell me about streaming data", stream=True)
        chunks = list(response) # Consume the stream by converting the iterator to a list
        
        assert len(chunks) > 0
        assert "".join(chunks)  # Make sure we got some content
        print(f"Streaming test passed")
    except Exception as e:
        print(f"Error testing streaming: {str(e)}")
        traceback.print_exc(file=sys.stdout)
        raise

# Test error handling for invalid options
@requires_api_key
@pytest.mark.parametrize(
    "option_name,invalid_value,expected_exception",
    [
        ("temperature", 3.0, ValueError),  # Outside valid range
        ("top_p", 2.0, ValueError),  # Outside valid range
        ("top_k", 3000, ValueError),  # Outside valid range
        ("search_recency_filter", "invalid_filter", ValueError),  # Invalid value
    ],
)
def test_invalid_options(option_name, invalid_value, expected_exception):
    """Test error handling for invalid option values."""
    try:
        print(f"\nTesting invalid option: {option_name}={invalid_value}")
        model = llm.get_model("sonar-pro")
        print(f"Model instance: {model}")
        with pytest.raises(expected_exception):
            # Create a prompt object with the invalid option
            prompt = llm.Prompt(
                "This should fail",
                model=model,
                options={option_name: invalid_value}
            )
            model.execute(prompt)
        print(f"Invalid option test passed for {option_name}={invalid_value}")
    except Exception as e:
        print(f"Error testing invalid option {option_name}={invalid_value}: {str(e)}")
        traceback.print_exc(file=sys.stdout)
        raise

@requires_api_key
def test_citations_handling():
    """Test that citations are properly handled for online models."""
    try:
        print(f"\nTesting citations handling")
        model = llm.get_model("sonar-pro-online")
        print(f"Model instance: {model}")
        response = model.prompt(
            "What are the latest developments in generative AI?", 
            stream=False
        )
        assert response is not None
        text = response.text()
        assert len(text) > 0
        print(f"Citations test passed")
    except Exception as e:
        print(f"Error testing citations: {str(e)}")
        traceback.print_exc(file=sys.stdout)
        raise

# Only run OpenRouter test if the key is set
@pytest.mark.skipif(
    not os.environ.get("LLM_OPENROUTER_KEY"),
    reason="LLM_OPENROUTER_KEY environment variable not set",
)
def test_openrouter_access():
    """Test accessing models through OpenRouter."""
    try:
        print(f"\nTesting OpenRouter access")
        model = llm.get_model("sonar-small")
        print(f"Model instance: {model}")
        
        # Create a model-specific prompt
        prompt = llm.Prompt(
            "Fun facts about pelicans",
            model=model,
            options={"use_openrouter": True}
        )
        response = model.execute(prompt)
        
        assert response is not None
        assert len(response.text()) > 0
        print(f"OpenRouter test passed")
    except Exception as e:
        print(f"Error testing OpenRouter: {str(e)}")
        traceback.print_exc(file=sys.stdout)
        raise
