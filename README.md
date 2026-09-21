# llm-perplexity

[![PyPI](https://img.shields.io/pypi/v/llm-perplexity.svg)](https://pypi.org/project/llm-perplexity/)
[![Changelog](https://img.shields.io/github/v/release/hex/llm-perplexity?include_prereleases&label=changelog)](https://github.com/hex/llm-perplexity/releases)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/hex/llm-perplexity/blob/main/LICENSE)

LLM plugin for Perplexity's Agent API

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

## Available Models

Perplexity's Sonar chat completions API will be supported until September 27, 2026. This plugin sends prompts through Perplexity's Agent API instead. The four Sonar model ids are still here, and each one selects an Agent API preset:

- **sonar** selects the `fast` preset
- **sonar-pro** selects the `low` preset
- **sonar-reasoning-pro** selects the `medium` preset
- **sonar-deep-research** selects the `high` preset

Perplexity chooses the model behind each preset and can change it. Check the `model` field in the logged response JSON (`llm logs --json`) to see which model actually answered.

Run prompts like this:

```bash
# Flagship model
llm -m sonar-pro 'Latest AI research'

# Base model
llm -m sonar 'Fun facts about walruses'

# Research and reasoning models
llm -m sonar-deep-research 'Complex research question'
llm -m sonar-reasoning-pro 'Problem solving task'
```

### Advanced Options

The plugin supports these parameters to customize model behavior. Some underlying models ignore `temperature` and `top_p`. Setting both is an error.

```bash
# Control randomness (0.0 up to but not including 2.0, higher = more random)
llm -m sonar-pro --option temperature 0.7 'Generate creative ideas'

# Nucleus sampling threshold (alternative to temperature)
llm -m sonar-pro --option top_p 0.9 'Generate varied responses'

# Limit response length
llm -m sonar-pro --option max_tokens 500 'Summarize this article'

# Suppress citations section and discourage inline [n] markers
llm -m sonar-pro --option include_citations false 'Latest AI research'

# Filter search results by domain
llm -m sonar-pro --option search_domain_filter 'arxiv.org,nature.com' 'Recent AI papers'

# Exclude a domain from search results
llm -m sonar-pro --option search_domain_filter '-example.org' 'Recent AI papers'

# Filter search results by recency (hour, day, week, month, year)
llm -m sonar-pro --option search_recency_filter year 'Major events'

# Control reasoning effort (minimal, low, medium, high)
llm -m sonar-reasoning-pro --option reasoning_effort high 'Solve this complex math problem'

# Control how much search context the model retrieves (low, medium, high)
llm -m sonar-pro --option search_context_size low 'Latest AI research'
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

`image_path` sends the image as an `input_image` part alongside the prompt text. `llm`'s own `-a` attachment flag is not supported yet. In a conversation, the plugin sends only the current turn's image and does not re-send images from earlier turns.

Note: Only certain Perplexity models support image inputs. The plugin forwards any `image/*` file; PNG, JPEG, and GIF are what Perplexity's models accept, not a plugin restriction.

## Changes in 2026.9.0

This release moved the plugin from Perplexity's Sonar chat completions API to the Agent API. The Agent API does not support 13 chat completions options. Passing any of them is now an error:

- `top_k`
- `stream`
- `presence_penalty`
- `frequency_penalty`
- `search_type`
- `search_mode`
- `disable_search`
- `search_language_filter`
- `return_images`
- `return_related_questions`
- `language_preference`
- `stop`
- `use_openrouter`

Anyone using `use_openrouter` to route through OpenRouter should install the [llm-openrouter](https://github.com/simonw/llm-openrouter) plugin instead.

Conversations logged before this upgrade still work with `llm -c`. A default saved with `llm models options set`, or an alias or template, that carries one of the options above now fails; check with `llm models options show <model>` and clear it.

`--key` now works. Earlier the plugin silently ignored it.

The logged response JSON (`llm logs --json`) now has the Agent API shape. Sources sit under an `output` item of type `search_results`, each with `title`, `url`, `date`, `last_updated`, and `snippet`. The old top-level `citations`, `search_results`, and `choices` keys are gone.

A response Perplexity reports as failed now raises an error instead of returning empty text. The plugin sends `max_tokens` to the API as `max_output_tokens`; in testing, a response cut short by it still reported status `completed`.

This release also raises the minimum versions to `llm>=0.26`, `openai>=1.109.1`, and Python `>=3.10`.

## Development

Clone the repository and run the tests:

```bash
git clone https://github.com/hex/llm-perplexity.git
cd llm-perplexity
uv run pytest
```

The suite runs offline against recorded cassettes and needs no API key.

To re-record a cassette against the live API, delete it under `tests/cassettes/` and run:

```bash
LLM_PERPLEXITY_KEY=your_perplexity_api_key uv run pytest --record-mode=once
```

This plugin was made after the [llm-claude-3](https://github.com/simonw/llm-claude-3) plugin by Simon Willison.
