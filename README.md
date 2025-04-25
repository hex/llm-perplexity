# llm-perplexity

[![PyPI](https://img.shields.io/pypi/v/llm-perplexity.svg)](https://pypi.org/project/llm-perplexity/)
[![Changelog](https://img.shields.io/github/v/release/hex/llm-perplexity?include_prereleases&label=changelog)](https://github.com/hex/llm-perplexity/releases)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/hex/llm-perplexity/blob/main/LICENSE)

LLM access to pplx-api 3 by Perplexity Labs

## Installation

Install this plugin in the same environment as [LLM](https://llm.datasette.io/).

```bash
llm install llm-perplexity
```

## Usage

First, set an [API key](https://www.perplexity.ai/settings/api) for Perplexity AI:

```bash
llm keys set perplexity
# Paste key here
```

Run `llm models` to list the models, and `llm models --options` to include a list of their options.

Run prompts like this:

### Standard Models

```bash
# Flagship model
llm -m sonar-pro 'Fun facts about AI'

# Base model
llm -m sonar 'Fun facts about walruses'
```

### Online Models with Web Search

```bash
# Flagship model with web search - for up-to-date information
llm -m sonar-pro-online 'Latest AI research in 2025'

# Filter search by recency - restrict to recent sources
llm -m sonar-pro-online --option search_recency_filter day 'Tech news today'

# Filter search by recency - specific time periods
llm -m sonar-pro-online --option search_recency_filter week 'Tech news this week'
llm -m sonar-pro-online --option search_recency_filter month 'Tech news this month'
llm -m sonar-pro-online --option search_recency_filter hour 'Very recent news'

# Filter search by domain - specify allowed domains
llm -m sonar-pro-online --option search_domain_filter github.com,arxiv.org 'LLM advancements'
```

### Other Available Models

```bash
# Research and reasoning models
llm -m sonar-deep-research 'Complex research question'
llm -m sonar-reasoning-pro 'Problem solving task'
llm -m sonar-reasoning 'Logical reasoning'
llm -m r1-1776 'Fun facts about seals'
```

### Advanced Options

The plugin supports various parameters to customize model behavior:

```bash
# Control randomness (0.0 to 2.0, higher = more random)
llm -m sonar-pro --option temperature 0.7 'Generate creative ideas'

# Nucleus sampling threshold (alternative to temperature)
llm -m sonar-pro --option top_p 0.9 'Generate varied responses'

# Token filtering (between 0 and 2048)
llm -m sonar-pro --option top_k 40 'Generate focused content'

# Limit response length
llm -m sonar-pro --option max_tokens 500 'Summarize this article'

# Return related questions
llm -m sonar-pro-online --option return_related_questions true 'How does quantum computing work?'
```

### Using Images with Perplexity

The plugin supports sending images to Perplexity models for analysis (multi-modal input):

```bash
# Analyze an image with Perplexity
llm -m sonar-pro --option image_path /path/to/your/image.jpg 'What can you tell me about this image?'

# Ask specific questions about an image
llm -m sonar-pro --option image_path /path/to/screenshot.png 'What text appears in this screenshot?'

# Multi-modal conversation with an image
llm -m sonar-pro --option image_path /path/to/diagram.png 'Explain the process shown in this diagram'
```

Note: Only certain Perplexity models support image inputs. Currently the following formats are supported: PNG, JPEG, and GIF.

### OpenRouter Access

You can also access these models through OpenRouter. First install the OpenRouter plugin:

```bash
llm install llm-openrouter
```

Then set your OpenRouter API key:

```bash
llm keys set openrouter
```

Use the `--option use_openrouter true` flag to route requests through OpenRouter:

```bash
llm -m sonar-pro --option use_openrouter true 'Fun facts about pelicans'
```

## Development

To set up this plugin locally, first checkout the code. Then create a new virtual environment:

```bash
cd llm-perplexity
python3 -m venv venv
source venv/bin/activate
```

Now install the dependencies and test dependencies:

```bash
llm install -e '.[test]'
```

### Running Tests

The test suite is comprehensive and tests all example commands from the documentation with actual API calls.

Before running tests, you need to set up your environment variables:

1. Copy the `.env.example` file to `.env`:
   ```bash
   cp .env.example .env
   ```

2. Edit the `.env` file and add your Perplexity API key:
   ```
   LLM_PERPLEXITY_KEY=your_perplexity_api_key_here
   ```

3. (Optional) If you want to test OpenRouter integration, also add your OpenRouter API key:
   ```
   LLM_OPENROUTER_KEY=your_openrouter_api_key_here
   ```

4. Install the package and test dependencies using one of these methods:

   **Using the setup script:**
   ```bash
   ./setup.sh
   ```

   **Using make:**
   ```bash
   make setup
   ```

   **Manually:**
   ```bash
   pip install -e .
   pip install pytest python-dotenv pillow
   ```

Run the tests with pytest:

```bash
# Run all tests
pytest test_llm_perplexity.py

# Using make
make test

# Run a specific test
pytest test_llm_perplexity.py::test_standard_models
```

Note: Running the full test suite will make real API calls to Perplexity, which may incur costs depending on your account plan.

This plugin was made after the [llm-claude-3](https://github.com/simonw/llm-claude-3) plugin by Simon Willison.