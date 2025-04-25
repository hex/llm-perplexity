#!/usr/bin/env python
"""
Script to check which Perplexity models are available with the current API key.
"""
import os
import sys
import llm
import dotenv
from rich.console import Console
from rich.table import Table

# Load environment variables from .env file
dotenv.load_dotenv()

# Check if API key is set
api_key = os.environ.get("LLM_PERPLEXITY_KEY")
if not api_key:
    print("Error: LLM_PERPLEXITY_KEY environment variable not set.")
    print("Please set it in your .env file or export it in your shell.")
    sys.exit(1)

# Models to check
MODELS = [
    "sonar-pro",
    "sonar",
    "sonar-pro-online",
    "sonar-deep-research",
    "sonar-reasoning-pro",
    "sonar-reasoning",
    "r1-1776"
]

console = Console()
table = Table(title="Perplexity Model Availability")
table.add_column("Model", style="cyan")
table.add_column("Available", style="green")
table.add_column("Error", style="red")

# Check each model
console.print("Checking model availability...\n")

for model_id in MODELS:
    try:
        model = llm.get_model(model_id)
        # Simple ping with minimal token usage
        response = model.prompt("Hi", stream=False)
        text = response.text().strip()
        status = "✅ Yes" if text else "❌ No response"
        error = ""
    except Exception as e:
        status = "❌ No"
        error = str(e)
    
    table.add_row(model_id, status, error)

console.print(table)
console.print("\nNote: Some models may require specific subscription tiers.")
console.print("If you're having trouble with a specific model, check your API key permissions.") 