.PHONY: setup test test-standard test-invalid test-all

setup:
	pip install -e .
	pip install pytest python-dotenv pillow rich

test-standard:
	pytest test_llm_perplexity.py::test_standard_models -v

test-invalid:
	pytest test_llm_perplexity.py::test_invalid_options -v

test-all:
	pytest test_llm_perplexity.py -v

# Default test target runs all tests
test: test-all
