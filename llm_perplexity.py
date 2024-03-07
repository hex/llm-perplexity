import llm
from openai import OpenAI
from pydantic import Field
from typing import List

@llm.hookimpl
def register_models(register):
    # https://docs.perplexity.ai/docs/model-cards
    register(Perplexity("sonar-small-chat"), aliases=("pp-small-chat",))
    register(Perplexity("sonar-small-online"), aliases=("pp-small-online",))
    register(Perplexity("sonar-medium-chat"), aliases=("pp-medium-chat",))
    register(Perplexity("sonar-medium-online"), aliases=("pp-medium-online",))
    register(Perplexity("codellama-70b-instruct"), aliases=("pp-70b-instruct",))
    register(Perplexity("mistral-7b-instruct"), aliases=("pp-7b-instruct",))
    register(Perplexity("mixtral-8x7b-instruct"), aliases=("pp-8x7b-instruct",))

class Perplexity(llm.Model):
    needs_key = "perplexity"
    key_env_var = "PERPLEXITY_API_KEY"
    model_id = "perplexity"
    can_stream = True

    class Options(llm.Options):
        max_tokens: int = Field(
            description="The maximum number of tokens to generate before stopping",
            default=4096,
        )

    def __init__(self, model_id):
        self.model_id = model_id

    def build_messages(self, prompt, conversation) -> List[dict]:
        messages = []
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

    def execute(self, prompt, stream, conversation):
        client = OpenAI(api_key=self.get_key(), base_url="https://api.perplexity.ai")

        kwargs = {
            "model": self.model_id,
            "messages": self.build_messages(prompt, conversation),
            "stream": stream,
            "max_tokens": prompt.options.max_tokens,
        }

        if prompt.system:
            kwargs["system"] = prompt.system

        if stream:
            with client.chat.completions.create(**kwargs) as stream:
                for text in stream:
                    yield text.choices[0].delta.content
        else:
           completion = client.chat.completions.create(**kwargs)
           yield completion.choices[0].message.content

    def __str__(self):
        return f"Perplexity: {self.model_id}"