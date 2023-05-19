from dotenv import load_dotenv
import tiktoken
import openai
import os

load_dotenv()

openai.api_key = os.getenv("OPENAI_API")


class Preprocess:
    def __init__(self, model: str):
        self.encoder = tiktoken.encoding_for_model(model_name=model)
        self.model = model

    def get_response(self, max_tokens: int = 512):
        response = openai.ChatCompletion.create(
            model=self.chat_model,
            messages=self.messages,
            temperature=0.4,
            max_tokens=max_tokens,
            top_p=1.0,
            frequency_penalty=0,
            presence_penalty=0,
        )
        self.messages.append(dict(response["choices"][0]["message"]))
        return self.messages[-1]["content"]
