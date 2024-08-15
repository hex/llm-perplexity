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

```bash
llm -m sonar-small 'Fun facts about pelicans'
llm -m sonar-large 'Fun facts about walruses'
llm -m sonar-huge 'Fun facts about whales'
llm -m sonar-small-chat 'Fun facts about wolves'
llm -m sonar-large-chat 'Fun facts about foxes'
llm -m llama-3.1-8b-instruct 'Fun facts about lemurs'
llm -m llama-3.1-70b-instruct 'Fun facts about coyotes'
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
