#!/bin/bash
# Setup script for llm-perplexity testing

echo "Installing llm-perplexity in development mode..."
pip install -e .

echo "Installing test dependencies..."
pip install pytest python-dotenv pillow

echo "Setup complete! You can now run the tests with:"
echo "pytest test_llm_perplexity.py"
echo ""
echo "To run specific tests:"
echo "pytest test_llm_perplexity.py::test_standard_models"
echo "pytest test_llm_perplexity.py::test_invalid_options"
