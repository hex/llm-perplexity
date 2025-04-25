#!/usr/bin/env python
"""
Script to selectively run pytest tests for models that are available with the current API key.
"""
import os
import sys
import subprocess
import dotenv
import llm

# Load environment variables from .env file
dotenv.load_dotenv()

# Check if API key is set
api_key = os.environ.get("LLM_PERPLEXITY_KEY")
if not api_key:
    print("Error: LLM_PERPLEXITY_KEY environment variable not set.")
    print("Please set it in your .env file or export it in your shell.")
    sys.exit(1)

print("Checking model availability...")
working_models = []

# Models to check
ALL_MODELS = [
    "sonar-pro",
    "sonar",
    "sonar-pro-online",
    "sonar-deep-research",
    "sonar-reasoning-pro",
    "sonar-reasoning",
    "r1-1776"
]

# Check each model by attempting a simple prompt
for model_id in ALL_MODELS:
    print(f"Checking {model_id}...", end="", flush=True)
    try:
        model = llm.get_model(model_id)
        # Simple ping with minimal token usage and a short timeout
        response = model.prompt("Hi", stream=False)
        text = response.text().strip()
        if text:
            print(" ✅ Working")
            working_models.append(model_id)
        else:
            print(" ❌ No response")
    except Exception as e:
        print(f" ❌ Error: {str(e)}")

if not working_models:
    print("\nNo working models found. Please check your API key permissions.")
    sys.exit(1)

# Build pytest expression to run tests only for working models
expressions = []

# Standard models test
standard_models = list(set(working_models) & set(["sonar-pro", "sonar-small", "sonar-medium", "sonar"]))
if standard_models:
    patterns = [f"test_standard_models[{model}-" for model in standard_models]
    expressions.extend(patterns)

# Online models test - only if we have online models working
online_models = list(set(working_models) & set(["sonar-pro-online", "sonar-small-online", "sonar-medium-online"]))
if online_models:
    patterns = [f"test_online_models_and_filters[{model}-" for model in online_models]
    expressions.extend(patterns)

# Other models test
other_models = list(set(working_models) & set([
    "sonar-deep-research", "sonar-reasoning-pro", "sonar-reasoning",
    "mistral-7b", "codellama-34b", "llama-2-70b", "r1-1776"
]))
if other_models:
    patterns = [f"test_other_models[{model}-" for model in other_models]
    expressions.extend(patterns)

# Add the remaining tests that don't depend on specific models
if "sonar-pro" in working_models:
    expressions.extend([
        "test_advanced_options",
        "test_image_analysis",
        "test_streaming_response",
        "test_invalid_options",
    ])

if "sonar-pro-online" in working_models:
    expressions.append("test_citations_handling")

# Check if OpenRouter key is set and add that test if so
if os.environ.get("LLM_OPENROUTER_KEY") and "sonar-small" in working_models:
    expressions.append("test_openrouter_access")

# Build the pytest command
pytest_expression = " or ".join(expressions)
command = f"python -m pytest test_llm_perplexity.py::'{pytest_expression}' -v"

print("\nRunning selective tests for working models:")
print(f"Working models: {', '.join(working_models)}")
print(f"Command: {command}\n")

# Run the pytest command
subprocess.run(command, shell=True) 