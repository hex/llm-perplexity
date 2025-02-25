import llm
from llm.utils import (
    remove_dict_none_values,
    simplify_usage_dict,
)
from openai import OpenAI
from pydantic import Field, field_validator, model_validator
from typing import Optional, List


@llm.hookimpl
def register_models(register):
    # https://docs.perplexity.ai/guides/model-cards
    register(Perplexity("sonar-deep-research"))
    register(Perplexity("sonar-reasoning-pro"))
    register(Perplexity("sonar-reasoning"))
    register(Perplexity("sonar-pro"))
    register(Perplexity("sonar"))
    register(Perplexity("r1-1776"))

class PerplexityOptions(llm.Options):
    max_tokens: Optional[int] = Field(
        description="The maximum number of completion tokens returned by the API. The total number of tokens requested in max_tokens plus the number of prompt tokens sent in messages must not exceed the context window token limit of model requested. If left unspecified, then the model will generate tokens until either it reaches its stop token or the end of its context window",
        default=None,
    )

    temperature: Optional[float] = Field(
        description="The amount of randomness in the response, valued between 0 inclusive and 2 exclusive. Higher values are more random, and lower values are more deterministic",
        default=1,
    )

    top_p: Optional[float] = Field(
        description="The nucleus sampling threshold, valued between 0 and 1 inclusive. For each subsequent token, the model considers the results of the tokens with 'top_p' probability mass. We recommend either altering 'top_k' or 'top_p', but not both",
        default=None,
    )

    top_k: Optional[int] = Field(
        description="The number of tokens to keep for highest 'top-k' filtering, specified as an integer between 0 and 2048 inclusive. If set to 0, 'top-k' filtering is disabled. We recommend either altering 'top_k' or 'top_p', but not both",
        default=None,
    )

    stream: Optional[bool] = Field(
        description="Determines whether or not to incrementally stream the response with server-sent events with 'content-type: text/event-stream'",
        default=True,
    )

    presence_penalty: Optional[float] = Field(
        description="A value between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics. Incompatible with 'frequency_penalty'",
        default=None,
    )

    frequency_penalty: Optional[float] = Field(
        description="A multiplicative penalty greater than 0. Values greater than 1.0 penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim. A value of 1.0 means no penalty. Incompatible with 'presence_penalty'",
        default=None,
    )

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, temperature):
        if not (0.0 <= temperature < 2.0):
            raise ValueError("temperature must be in range 0-2")
        return temperature

    @field_validator("top_p")
    @classmethod
    def validate_top_p(cls, top_p):
        if top_p is not None and not (0.0 <= top_p <= 1.0):
            raise ValueError("top_p must be in range 0.0-1.0")
        return top_p

    @field_validator("top_k")
    @classmethod
    def validate_top_k(cls, top_k):
        if top_k is not None and top_k <= 0 or top_k > 2048:
            raise ValueError("top_k must be in range 0-2048")
        return top_k

    @model_validator(mode="after")
    def validate_temperature_top_p(self):
        if self.temperature != 1.0 and self.top_p is not None:
            raise ValueError("Only one of temperature and top_p can be set")
        return self


class Perplexity(llm.Model):
    needs_key = "perplexity"
    key_env_var = "LLM_PERPLEXITY_KEY"
    model_id = "perplexity"
    can_stream = True
    base_url = "https://api.perplexity.ai"

    class Options(PerplexityOptions):
        use_openrouter: Optional[bool] = Field(
            description="Whether to use OpenRouter API instead of direct Perplexity API",
            default=False,
        )

    def __init__(self, model_id):
        self.model_id = model_id

    @staticmethod 
    def combine_chunks(chunks: List) -> dict:
        content = ""
        role = None
        finish_reason = None
        # If any of them have log probability, we're going to persist
        # those later on
        logprobs = []
        usage = {}
        citations = {}

        for item in chunks:
            if item.usage:
                usage = item.usage.dict()
                
            if item.citations:
                citations = item.citations

            for choice in item.choices:
                if choice.logprobs and hasattr(choice.logprobs, "top_logprobs"):
                    logprobs.append(
                        {
                            "text": choice.text if hasattr(choice, "text") else None,
                            "top_logprobs": choice.logprobs.top_logprobs,
                        }
                    )

                if not hasattr(choice, "delta"):
                    content += choice.text
                    continue
                role = choice.delta.role
                if choice.delta.content is not None:
                    content += choice.delta.content
                if choice.finish_reason is not None:
                    finish_reason = choice.finish_reason
        

        # Imitations of the OpenAI API may be missing some of these fields
        combined = {
            "content": content,
            "role": role,
            "finish_reason": finish_reason,
            "usage": usage,
            "citations": citations,
        }
        if logprobs:
            combined["logprobs"] = logprobs
        if chunks:
            for key in ("id", "object", "model", "created", "index"):
                value = getattr(chunks[0], key, None)
                if value is not None:
                    combined[key] = value

        return combined

    def build_messages(self, prompt, conversation) -> List[dict]:
        messages = []
        if prompt.system:
            messages.append({"role": "system", "content": prompt.system})
        if conversation:
            for response in conversation.responses:
                messages.extend(
                    [
                        {
                            "role": "user",
                            "content": response.prompt.prompt,
                        },
                        {"role": "assistant", "content": response.text()},
                    ]
                )
        messages.append({"role": "user", "content": prompt.prompt})
        return messages

    def set_usage(self, response, usage):
        if not usage:
            return
    
        input_tokens = usage.pop("prompt_tokens")
        output_tokens = usage.pop("completion_tokens")
        usage.pop("total_tokens")
        response.set_usage(
            input=input_tokens, output=output_tokens, details=simplify_usage_dict(usage)
        )

    def execute(self, prompt, stream, response, conversation):
        if prompt.options.use_openrouter:
            if not any(p["name"] == "llm-openrouter" for p in llm.get_plugins()):
                raise llm.ModelError(
                    "OpenRouter support requires the llm-openrouter plugin. "
                    "Install it with: llm install llm-openrouter"
                )
            api_key = llm.get_key("openrouter", "LLM_OPENROUTER_KEY")
            base_url = "https://openrouter.ai/api/v1"
            model_id = (
                f"meta-llama/llama-3.3-70b-instruct"
                if self.model_id == "llama-3.3-70b-instruct"
                else f"perplexity/{self.model_id}"
            )
        else:
            api_key = self.get_key()
            base_url = self.base_url
            model_id = self.model_id

        client = OpenAI(api_key=api_key, base_url=base_url)

        kwargs = {
            "model": model_id,
            "messages": self.build_messages(prompt, conversation),
            "stream": stream,
            "max_tokens": prompt.options.max_tokens or None,
        }

        if prompt.options.top_p:
            kwargs["top_p"] = prompt.options.top_p
        else:
            kwargs["temperature"] = prompt.options.temperature

        if prompt.options.top_k:
            kwargs["top_k"] = prompt.options.top_k

        if stream:
            completion = client.chat.completions.create(**kwargs)
            chunks = []
            usage = None
            citations = None

            for chunk in completion:
                chunks.append(chunk)
                if hasattr(chunk, "usage") and chunk.usage:
                    usage = chunk.usage.model_dump()
                if hasattr(chunk, "citations") and chunk.citations:
                     citations = chunk.citations   
                try:
                    content = chunk.choices[0].delta.content
                except IndexError:
                    content = None
                if content is not None:
                    yield content
            response.response_json = remove_dict_none_values(Perplexity.combine_chunks(chunks))
            
            if citations:
                yield "\n\n## Citations:\n"
                for i, citation in enumerate(citations, 1):
                    yield f"[{i}] {citation}\n"

        else:
            completion = client.chat.completions.create(**kwargs)
            response.response_json = remove_dict_none_values(completion.model_dump())
            usage = completion.usage.model_dump()
            yield completion.choices[0].message.content
            if hasattr(completion, "citations") and completion.citations:
                yield "\n\n## Citations:\n"
                for i, citation in enumerate(completion.citations, 1):
                    yield f"[{i}] {citation}\n"
        self.set_usage(response, usage)
        response._prompt_json = {"messages": kwargs["messages"]}

    def __str__(self):
        return f"Perplexity: {self.model_id}"
