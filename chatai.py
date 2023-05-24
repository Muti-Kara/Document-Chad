from dotenv import load_dotenv
import backoff
import openai
import os

load_dotenv()

openai.api_key = os.getenv("OPENAI_API")


class Embedder:
    def __init__(self, model: str):
        self.model = model

    # @backoff.on_exception(backoff.expo, Exception, max_tries=10)
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
                    info_1=srcs[0],
                    info_2=srcs[1],
                    info_3=srcs[2],
                    info_4=srcs[3],
                    question=question
                )
            }
        )

    # @backoff.on_exception(backoff.expo, Exception, max_tries=10)
    def get_response(self):
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
            return self.get_response()

        self.messages.append(response["choices"][0]["message"])
        return RESPONSE_FORMAT.format(answer=self.messages[-1]["content"])

    def clear_all(self):
        self.messages = [
            {
                "role": "system",
                "content": CHAT_PROMPT
            }
        ]


PREPROCESS_PROMPT = """
You are a text processor. When user gives a text you should only response with the corrected version. The text you will receive will not contain any white-space at all, so you need to insert them. Also It may have some markup symbols and you should exclude those. If you find spelling errors fix them too.\n

Input Format:
{raw text}

Output Format:
{corrected text}
"""
CHAT_PROMPT = """
You are an AI assistant that will answer user's questions by looking at the information user will give. User will give the information you will use in 4 blocks with and the question he/she asked.\n

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
{answer}
"""
