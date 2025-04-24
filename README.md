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

# Lightweight model
llm -m sonar-small 'Fun facts about sharks'

# Mid-sized model
llm -m sonar-medium 'Fun facts about plums'

# Base model
llm -m sonar 'Fun facts about walruses'
```

### Online Models with Web Search

```bash
# Flagship model with web search - for up-to-date information
llm -m sonar-pro-online 'Latest AI research in 2025'

# Filter search by recency - restrict to recent sources
llm -m sonar-small-online --option search_recency_filter week 'Tech news this week'

# Filter search by domain - specify allowed domains
llm -m sonar-medium-online --option search_domain_filter github.com,arxiv.org 'LLM advancements'
```

### Other Available Models

```bash
# Legacy models
llm -m sonar-deep-research 'Complex research question'
llm -m sonar-reasoning-pro 'Problem solving task'
llm -m sonar-reasoning 'Logical reasoning'

# Open-source models
llm -m mistral-7b 'Simple general task'
llm -m codellama-34b 'Write a Python function'
llm -m llama-2-70b 'Generate creative content'
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

# Return images in response (if model supports it)
llm -m sonar-pro-online --option return_images true 'Show me diagrams of neural networks'

# Return related questions
llm -m sonar-pro-online --option return_related_questions true 'How does quantum computing work?'
```

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
llm -m sonar-small --option use_openrouter true 'Fun facts about pelicans'
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

This plugin was made after the [llm-claude-3](https://github.com/simonw/llm-claude-3) plugin by Simon Willison.
