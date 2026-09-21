# ABOUTME: LLM plugin that sends prompts to Perplexity's Agent API.
# ABOUTME: Maps the Sonar model ids to Agent API presets and formats search citations.
import base64
import mimetypes
import re
from importlib.metadata import PackageNotFoundError, version

import llm
from llm.utils import (
    remove_dict_none_values,
    simplify_usage_dict,
)
from openai import APIError, OpenAI
from pydantic import Field, field_validator, model_validator
from typing import Optional, List, Literal

CITATIONS_HEADING = "\n\n## Citations:\n"


CITATIONS_FOOTER_RE = re.compile(re.escape(CITATIONS_HEADING) + r"(?:\[\d+\] [^\n]*\n?)+\Z")


def strip_citations(text: str) -> str:
    """Remove the citations section that format_citations appends to response text."""
    return CITATIONS_FOOTER_RE.sub("", text)


# Perplexity's suggested Agent API preset for each Sonar model id
# https://docs.perplexity.ai/docs/agent-api/migrate-from-sonar/how-to
PRESETS = {
    "sonar": "fast",
    "sonar-pro": "low",
    "sonar-reasoning-pro": "medium",
    "sonar-deep-research": "high",
}


def web_search_tool(options) -> Optional[dict]:
    """Build the web_search tool entry, or None when no search option is set."""
    filters = {}
    if options.search_domain_filter:
        filters["search_domain_filter"] = [
            d.strip() for d in options.search_domain_filter.split(",") if d.strip()
        ]
    if options.search_recency_filter:
        filters["search_recency_filter"] = options.search_recency_filter

    if not filters and not options.search_context_size:
        return None

    tool: dict = {"type": "web_search"}
    if options.search_context_size:
        tool["search_context_size"] = options.search_context_size
    if filters:
        tool["filters"] = filters
    return tool


def integration_header() -> str:
    """Value of the X-Pplx-Integration header, in Perplexity's name/version form."""
    try:
        return f"llm-perplexity/{version('llm-perplexity')}"
    except PackageNotFoundError:
        return "llm-perplexity/dev"


def search_results(response_data: dict) -> List[dict]:
    """Collect the sources from every search_results item of an Agent API response."""
    return [
        result
        for item in response_data.get("output") or []
        if item.get("type") == "search_results"
        for result in item.get("results") or []
    ]


def failure_message(response_data: dict) -> Optional[str]:
    """Describe why an Agent API response failed, or None when it did not fail."""
    if response_data.get("status") != "failed":
        return None
    error = response_data.get("error") or {}
    return error.get("message") or "the response failed without an error message"


@llm.hookimpl
def register_models(register):
    for model_id in PRESETS:
        register(Perplexity(model_id))

# Options the Agent API has no equivalent for, with the value llm logged for
# each when the user had not set it. Logged conversations still carry them.
UNSUPPORTED_OPTION_DEFAULTS = {
    "top_k": None,
    "stream": True,
    "presence_penalty": None,
    "frequency_penalty": None,
    "search_type": None,
    "search_mode": None,
    "disable_search": None,
    "search_language_filter": None,
    "return_images": None,
    "return_related_questions": False,
    "language_preference": None,
    "stop": None,
    "use_openrouter": False,
}


def unsupported_option_message(name: str) -> str:
    """Tell the user an option is unsupported and where the change is documented."""
    message = (
        f"{name} is not supported by Perplexity's Agent API. "
        'See "Changes in 2026.9.0" in the llm-perplexity README.'
    )
    if name == "use_openrouter":
        message += " To route through OpenRouter, install the llm-openrouter plugin."
    return message


class PerplexityOptions(llm.Options):
    max_tokens: Optional[int] = Field(
        description="The maximum number of output tokens the model may generate.",
        default=None,
    )

    temperature: Optional[float] = Field(
        description="The amount of randomness in the response, valued between 0 inclusive and 2 exclusive. Some underlying models ignore it.",
        default=None,
    )

    top_p: Optional[float] = Field(
        description="The nucleus sampling threshold, valued between 0 and 1 inclusive. Some underlying models ignore it.",
        default=None,
    )

    search_recency_filter: Optional[str] = Field(
        description="Filter search results by time period. Options: 'hour', 'day', 'week', 'month', 'year'.",
        default=None,
    )

    search_domain_filter: Optional[str] = Field(
        description="Comma-separated list of domains to search. Prefix a domain with '-' to exclude it. Up to 20 entries.",
        default=None,
    )

    search_context_size: Optional[Literal["low", "medium", "high"]] = Field(
        description="How much search context is retrieved for the model: 'low', 'medium' or 'high'.",
        default=None,
    )

    reasoning_effort: Optional[Literal["minimal", "low", "medium", "high"]] = Field(
        description="Control the computational effort for reasoning. Options: 'minimal', 'low', 'medium', 'high'.",
        default=None,
    )

    image_path: Optional[str] = Field(
        description="Path to an image file to include in the request. The image will be encoded as base64 and sent along with the text prompt.",
        default=None,
    )

    include_citations: Optional[bool] = Field(
        description="Include formatted citations section in the text output (does not affect JSON response)",
        default=True,
    )

    @model_validator(mode="before")
    @classmethod
    def discard_unsupported_options_left_at_their_default(cls, values):
        if not isinstance(values, dict):
            return values
        supported = {}
        for name, value in values.items():
            if name not in UNSUPPORTED_OPTION_DEFAULTS:
                supported[name] = value
            elif not (value is None or value is UNSUPPORTED_OPTION_DEFAULTS[name]):
                raise ValueError(unsupported_option_message(name))
        return supported

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, temperature):
        if temperature is not None and not (0.0 <= temperature < 2.0):
            raise ValueError("temperature must be at least 0 and below 2")
        return temperature

    @field_validator("top_p")
    @classmethod
    def validate_top_p(cls, top_p):
        if top_p is not None and not (0.0 <= top_p <= 1.0):
            raise ValueError("top_p must be in range 0.0-1.0")
        return top_p

    @field_validator("search_recency_filter")
    @classmethod
    def validate_search_recency_filter(cls, recency_filter):
        if recency_filter is not None and recency_filter not in ["hour", "day", "week", "month", "year"]:
            raise ValueError("search_recency_filter must be one of: 'hour', 'day', 'week', 'month', or 'year'")
        return recency_filter

    @field_validator("search_domain_filter")
    @classmethod
    def validate_search_domain_filter(cls, domain_filter):
        if domain_filter is not None:
            domains = [d.strip() for d in domain_filter.split(",")]
            if not all(d and "." in d for d in domains):
                raise ValueError("search_domain_filter must be a comma-separated list of valid domains")
            if len(domains) > 20:
                raise ValueError("search_domain_filter accepts at most 20 domains")
        return domain_filter

    @model_validator(mode="after")
    def validate_temperature_top_p(self):
        if self.temperature is not None and self.top_p is not None:
            raise ValueError("Only one of temperature and top_p can be set")
        return self


class Perplexity(llm.KeyModel):
    needs_key = "perplexity"
    key_env_var = "LLM_PERPLEXITY_KEY"
    can_stream = True
    base_url = "https://api.perplexity.ai/v1"

    class Options(PerplexityOptions):
        pass

    def __init__(self, model_id):
        self.model_id = model_id

    def build_input(self, prompt, conversation) -> List[dict]:
        past_turns = conversation.responses if conversation else []

        system = prompt.system or next(
            (turn.prompt.system for turn in past_turns if turn.prompt.system), None
        )
        system_message = "\n".join(filter(None, (
            system,
            "Do not include bracketed numeric citation markers like [1], [2]; integrate sources naturally without inline citation tokens."
            if prompt.options.include_citations is False else None
        )))

        items = []
        if system_message:
            items.append({"role": "system", "content": system_message})

        for turn in past_turns:
            items.append({"role": "user", "content": turn.prompt.prompt})
            items.append({"role": "assistant", "content": strip_citations(turn.text())})

        items.append({"role": "user", "content": self._user_content(prompt)})
        return items

    @staticmethod
    def _user_content(prompt):
        image_path = prompt.options.image_path
        if not image_path:
            return prompt.prompt

        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type:
            mime_type = "image/png"
        elif not mime_type.startswith("image/"):
            raise llm.ModelError(f"Error processing image: {image_path} is {mime_type}, not an image")

        try:
            with open(image_path, "rb") as img_file:
                encoded_image = base64.b64encode(img_file.read()).decode("utf-8")
        except OSError as e:
            raise llm.ModelError(f"Error processing image: {str(e)}")

        return [
            {"type": "input_text", "text": prompt.prompt},
            {"type": "input_image", "image_url": f"data:{mime_type};base64,{encoded_image}"},
        ]

    def build_request(self, prompt, conversation, stream: bool) -> dict:
        options = prompt.options
        request = {
            "input": self.build_input(prompt, conversation),
            "stream": stream,
            "extra_body": {"preset": PRESETS[self.model_id]},
        }
        if options.max_tokens is not None:
            request["max_output_tokens"] = options.max_tokens
        if options.temperature is not None:
            request["temperature"] = options.temperature
        if options.top_p is not None:
            request["top_p"] = options.top_p
        if options.reasoning_effort:
            request["reasoning"] = {"effort": options.reasoning_effort}

        tool = web_search_tool(options)
        if tool:
            request["tools"] = [tool]
        return request

    def set_usage(self, response, usage):
        if not usage:
            return

        details = {k: v for k, v in usage.items()
                   if k not in ("input_tokens", "output_tokens", "total_tokens")}
        response.set_usage(
            input=usage.get("input_tokens", 0),
            output=usage.get("output_tokens", 0),
            details=simplify_usage_dict(details),
        )

    @staticmethod
    def format_citations(citations, prefix=CITATIONS_HEADING) -> str:
        citation_texts = [
            " - ".join(filter(None, (citation.get("title"), citation.get("url"))))
            for citation in citations or []
        ]
        printable = list(filter(None, citation_texts))
        if not printable:
            return ""

        formatted = prefix
        for i, citation_text in enumerate(printable, 1):
            formatted += f"[{i}] {citation_text}\n"
        return formatted

    def execute(self, prompt, stream, response, conversation, key=None):
        client = OpenAI(
            api_key=self.get_key(key),
            base_url=self.base_url,
            default_headers={"X-Pplx-Integration": integration_header()},
        )
        request = self.build_request(prompt, conversation, stream)

        completed = None
        try:
            if stream:
                for event in client.responses.create(**request):
                    if event.type == "response.output_text.delta":
                        yield event.delta
                    elif event.type in ("response.completed", "response.incomplete", "response.failed"):
                        completed = event.response
            else:
                completed = client.responses.create(**request)
                yield completed.output_text
        except APIError as e:
            raise llm.ModelError(f"Perplexity API error: {e.message}") from e

        if completed is None:
            raise llm.ModelError("Perplexity ended the stream without a final response")

        # The openai SDK has no type for Perplexity's search_results output item,
        # so serialising with warnings on prints a pydantic warning per request
        response_data = completed.model_dump(warnings=False)
        response.response_json = remove_dict_none_values(response_data)
        response._prompt_json = {"input": request["input"]}

        failure = failure_message(response_data)
        if failure:
            raise llm.ModelError(f"Perplexity API error: {failure}")

        sources = search_results(response_data)
        if sources and prompt.options.include_citations:
            yield self.format_citations(sources)
        self.set_usage(response, response_data.get("usage"))

    def __str__(self):
        return f"Perplexity: {self.model_id}"
