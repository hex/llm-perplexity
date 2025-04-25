# Testing llm-perplexity

This directory contains tests for the llm-perplexity plugin using pytest and VCR.py.

## Running Tests

To run all tests:

```bash
python -m pytest
```

To run specific test files:

```bash
python -m pytest tests/test_perplexity.py
python -m pytest tests/test_conversation.py
```

To run specific test functions:

```bash
python -m pytest tests/test_perplexity.py::test_standard_model
```

## Test Environment

The tests use VCR.py to record HTTP interactions with the Perplexity API. This allows tests to run without making actual API calls after the first run, making them faster and more reliable.

### Recording Mode

To record new interactions or update existing ones, run:

```bash
# Record all tests
python -m pytest --vcr-record=all

# Record only new interactions (don't overwrite existing ones)
python -m pytest --vcr-record=new_episodes
```

### API Keys for Testing

The tests look for API keys in environment variables:

- `PYTEST_PERPLEXITY_API_KEY` - For Perplexity API tests
- `PYTEST_OPENROUTER_KEY` - For OpenRouter tests

If these variables are not set, the tests will use placeholder values and rely on pre-recorded interactions.

## Cassettes

VCR.py stores recorded API interactions in the `cassettes` directory. These files contain the HTTP request/response pairs but with sensitive information (like API keys) filtered out.

## How to Create New Tests

When adding a new test:

1. Create a test function in an appropriate test file
2. Run the test with `--vcr-record=all` to create cassettes
3. Verify the test works with the recorded cassettes
4. Commit the test and cassettes to version control

## Coverage

To see test coverage:

```bash
# Install coverage package if not already installed
pip install coverage

# Run tests with coverage
coverage run -m pytest

# Generate coverage report
coverage report
```
