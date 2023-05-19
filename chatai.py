from dotenv import load_dotenv
import openai
import os

load_dotenv()

openai.api_key = os.getenv("OPENAI_API")


class Embedder:
    def __init__(self, model: str):
        self.model = model

    def get_embedding(self, text: str):
        text = text.replace("\n", " ")
        return openai.Embedding.create(
            input=[text],
            model=self.model
        )['data'][0]['embedding']


class ChatBot:
    def __init__(self, model: str, answer_token: int):
        self.model = model
        self.max_tokens = answer_token
        self.clear_all()

    def add_question(self, question: str, srcs: list[str]):
        self.messages.append(
            {
                "role": "user",
                "content": QUESTION_FORMAT.format(
                    info_1=srcs[0] if len(srcs) > 0 else "",
                    info_2=srcs[1] if len(srcs) > 1 else "",
                    info_3=srcs[2] if len(srcs) > 2 else "",
                    info_4=srcs[3] if len(srcs) > 3 else "",
                    question=question
                )
            }
        )

    def get_response(self, srcs: list[str]):
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=self.messages,
                temperature=0.4,
                max_tokens=self.max_tokens,
                top_p=1.0,
                frequency_penalty=0,
                presence_penalty=0,
            )
        except Exception:
            question = self.messages[-1]
            self.clear_all()
            self.messages.append(question)
            return self.get_response(srcs)

        self.messages.append(response["choices"][0]["message"])
        return RESPONSE_FORMAT.format(
            answer=self.messages[-1]["content"],
            src1=srcs[0] if len(srcs) > 0 else "",
            src2=srcs[1] if len(srcs) > 1 else "",
            src3=srcs[2] if len(srcs) > 2 else "",
            src4=srcs[3] if len(srcs) > 3 else "",
        )

    def clear_all(self):
        self.messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            }
        ]


SYSTEM_PROMPT = """
You are an AI assistant that will answer user's questions by looking at
the information user will give.\nUser will give the information you will
use in 4 blocks with and the question he/she asked.\n

Input Format:
///\n
{information 1}\n
///\n
///\n
{information 2}\n
///\n
///\n
{information 3}\n
///\n
///\n
{information 4}\n
///\n
***\n
{question}\n
***\n
"""
QUESTION_FORMAT = """
///\n
{info_1}\n
///\n
///\n
{info_2}\n
///\n
///\n
{info_3}\n
///\n
///\n
{info_4}\n
///\n
***\n
{question}\n
***\n
"""
RESPONSE_FORMAT = """
ChatBots Answer: {answer}\n
Source 1: {src1}\n
Source 2: {src2}\n
Source 3: {src3}\n
Source 4: {src4}\n
"""
