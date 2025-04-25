import llm
import os
import pytest

# Set a default API key for tests if one isn't provided
PERPLEXITY_API_KEY = os.environ.get("PYTEST_PERPLEXITY_API_KEY", None) or "pplx-..."


@pytest.mark.vcr
def test_basic_conversation():
    """Test a basic conversation with a Perplexity model."""
    model = llm.get_model("sonar-small")
    model.key = model.key or PERPLEXITY_API_KEY
    
    conversation = model.conversation()
    
    # First message
    response1 = conversation.prompt("Name two types of sharks")
    text1 = response1.text()
    assert len(text1) > 0
    assert "shark" in text1.lower()
    
    # Follow-up message
    response2 = conversation.prompt("Which one is bigger?")
    text2 = response2.text()
    assert len(text2) > 0
    
    # Verify the model remembered context from the previous exchange
    # The model should mention sharks from the first response
    assert any(shark_name.lower() in text2.lower() for shark_name in text1.lower().split() if len(shark_name) > 3)


@pytest.mark.vcr
def test_conversation_with_system_prompt():
    """Test a conversation with a system prompt."""
    model = llm.get_model("sonar-small")
    model.key = model.key or PERPLEXITY_API_KEY
    
    # Create conversation with system prompt
    conversation = model.conversation(system="You are a marine biologist specializing in sharks.")
    
    # Ask a question
    response = conversation.prompt("Tell me about the hunting techniques of great white sharks")
    text = response.text()
    
    assert len(text) > 0
    assert any(term in text.lower() for term in ["hunt", "predator", "ambush", "prey"])
    
    # The response should show expertise about sharks
    assert "great white" in text.lower()


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_conversation():
    """Test an asynchronous conversation with a Perplexity model."""
    model = llm.get_async_model("sonar-small")
    model.key = model.key or PERPLEXITY_API_KEY
    
    conversation = model.conversation()
    
    # First message
    response1 = await conversation.prompt("Name three species of dolphins")
    text1 = await response1.text()
    assert len(text1) > 0
    assert "dolphin" in text1.lower()
    
    # Follow-up message
    response2 = await conversation.prompt("Which one is the most intelligent?")
    text2 = await response2.text()
    assert len(text2) > 0
    
    # Verify the model remembered context
    assert any(dolphin_name.lower() in text2.lower() for dolphin_name in text1.lower().split() if len(dolphin_name) > 3)
