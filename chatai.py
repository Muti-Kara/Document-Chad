from dotenv import load_dotenv
import openai
import os

load_dotenv()

openai.api_key = os.getenv("OPENAI_API")


class ChatBot:
    def __init__(self, chat_model: str, embedding_model:str, system_prompt: str):
        self.embedding_model = embedding_model
        self.chat_model = chat_model
        self.messages = []

    def get_embedding(self, text: str):
        text = text.replace("\n", " ")
        return openai.Embedding.create(
            input=[text],
            model=self.embedding_model
        )['data'][0]['embedding']

    def add_message(self, message: str, role: str):
        self.messages.append({"content": message, "role": role})

    def get_response(self):
        response = openai.ChatCompletion.create(
            model=self.chat_model,
            messages=self.messages,
            temperature=0.4,
            max_tokens=512,
            top_p=1.0,
            frequency_penalty=0,
            presence_penalty=0,
        )
        self.messages.append(dict(response["choices"][0]["message"]))
        return self.messages[-1]["content"]


CHAT_MODEL = "gpt-3.5-turbo"
EMBEDDING_MODEL = "text-embedding-ada-002"
SYSTEM_PROMPT = """
You are an AI assistant that will answer user's questions by
looking at the information user will give. User will give the
information you will use in 4 blocks with and the question he
or she asked with a formatting like this:

///
{information 1}
///
///
{information 2}
///
///
{information 3}
///
///
{information 4}
///
***
{question}
***
"""
QUESTION_PROMT = """
///
{info_1}
///
///
{info_2}
///
///
{info_3}
///
///
{info_4}
///
***
{question}
***
"""
