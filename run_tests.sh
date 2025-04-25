#!/bin/bash

# Script to run tests for llm-perplexity

# Make the script exit on any error
set -e

# Check if test dependencies are installed
if ! python -c "import pytest" &> /dev/null; then
    echo "Installing test dependencies..."
    llm install -e '.[test]'
fi

# Default mode is to use existing cassettes
RECORD_MODE="once"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --record)
            RECORD_MODE="all"
            shift
            ;;
        --new-episodes)
            RECORD_MODE="new_episodes"
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Run tests for llm-perplexity plugin"
            echo ""
            echo "Options:"
            echo "  --record         Record all API interactions, overwriting existing cassettes"
            echo "  --new-episodes   Record only new API interactions, preserving existing cassettes"
            echo "  --help           Display this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Run the tests
echo "Running tests with VCR record mode: $RECORD_MODE"
python -m pytest --vcr-record=$RECORD_MODE tests/

echo "All tests completed successfully!"
