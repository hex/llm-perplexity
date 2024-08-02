import llm
from openai import OpenAI
from pydantic import Field, field_validator, model_validator
from typing import Optional, List


@llm.hookimpl
def register_models(register):
    # https://docs.perplexity.ai/docs/model-cards
    register(Perplexity("llama-3.1-sonar-small-128k-online"))
    register(Perplexity("llama-3.1-sonar-large-128k-online"))
    register(Perplexity("llama-3.1-sonar-small-128k-chat"))
    register(Perplexity("llama-3.1-sonar-large-128k-chat"))
    register(Perplexity("llama-3.1-70b-instruct"))
    register(Perplexity("llama-3.1-8b-instruct"))
    
    # The following models will be deprecated on August 12 2024
    register(Perplexity("llama-3-sonar-small-32k-chat"))
    register(Perplexity("llama-3-sonar-small-32k-online"))
    register(Perplexity("llama-3-sonar-large-32k-chat"))
    register(Perplexity("llama-3-sonar-large-32k-online"))
    register(Perplexity("llama-3-8b-instruct"))
    register(Perplexity("llama-3-70b-instruct"))
    register(Perplexity("mixtral-8x7b-instruct"), aliases=("pp-8x7b-instruct",))


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
    key_env_var = "PERPLEXITY_API_KEY"
    model_id = "perplexity"
    can_stream = True

    class Options(PerplexityOptions): ...

    def __init__(self, model_id):
        self.model_id = model_id

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

    def execute(self, prompt, stream, response, conversation):
        client = OpenAI(api_key=self.get_key(), base_url="https://api.perplexity.ai")

        kwargs = {
            "model": self.model_id,
            "messages": self.build_messages(prompt, conversation),
            "stream": prompt.options.stream,
            "max_tokens": prompt.options.max_tokens or None,
        }

        if prompt.options.top_p:
            kwargs["top_p"] = prompt.options.top_p
        else:
            kwargs["temperature"] = prompt.options.temperature

        if prompt.options.top_k:
            kwargs["top_k"] = prompt.options.top_k

        if stream:
            with client.chat.completions.create(**kwargs) as stream:
                for text in stream:
                    yield text.choices[0].delta.content
        else:
            completion = client.chat.completions.create(**kwargs)
            yield completion.choices[0].message.content

    def __str__(self):
        return f"Perplexity: {self.model_id}"
