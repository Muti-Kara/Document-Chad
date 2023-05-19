import gradio as gr
import tiktoken
import time
import re

from chatai import ChatBot, Embedder
from vectors import VectorManager
from reader import Reader


CHAT_MODEL = "gpt-3.5-turbo"
CHAT_MODEL_TOKEN = 4097

EMBEDDING_MODEL = "text-embedding-ada-002"
EMBEDDING_TOKEN = 8191
EMBEDDING_DIMS = 1536

BATCH_LENGTH = 50
CHUNK_COUNT = 4

MAX_USER_TOKEN = 512
MAX_ANSWER_TOKEN = 512
MAX_CHUNK_SIZE = 256

encoder = tiktoken.encoding_for_model(
    model_name=EMBEDDING_MODEL
)
chat_bot = ChatBot(
    model=CHAT_MODEL,
    answer_token=MAX_ANSWER_TOKEN
)
embedder = Embedder(
    model=EMBEDDING_MODEL
)
vector_manager = VectorManager(
    index_name="qa-embedding",
    dims=EMBEDDING_DIMS,
    namespace="namespace"
)


def add_text(history, text):
    history = history + [(text, None)]
    return history, ""


def clear_conversation():
    vector_manager.clr()
    chat_bot.clear_all()
    return None


def add_file(history, file):
    reader = Reader(file.name)

    chunks = reader.split_chunks(
        model_name=EMBEDDING_MODEL,
        max_chunk_size=MAX_CHUNK_SIZE
    )

    batches = reader.batch_chunks(chunks, BATCH_LENGTH)

    embeddings = []
    for batch in batches:
        embeddings.append([embedder.get_embedding(chunk) for chunk in batch])
        if len(embeddings) < len(batches):
            time.sleep(60)

    for index in range(len(batches)):
        vector_manager.put(
            chunks=batches[index],
            embeddings=embeddings[index]
        )

    history.append((str(reader), None))
    return history


def bot(history):
    pattern = r"^File:\s\d+\ssentences in the document\.$"
    user_prompt = history[-1][0]
    if re.match(pattern=pattern, string=user_prompt):
        response = "I received, and digested your file :)"
    else:
        if len(encoder.encode(user_prompt)) > MAX_USER_TOKEN:
            response = "This is much longer message than I can answer :("
        else:
            embedding = embedder.get_embedding(user_prompt)
            relevant_chunks = vector_manager.get(embedding=embedding, top_k=CHUNK_COUNT)
            chat_bot.add_question(user_prompt, relevant_chunks)
            response = chat_bot.get_response(relevant_chunks)
    history[-1][1] = response
    return history


with gr.Blocks() as app:
    chat_box = gr.Chatbot().style(600)
    txt = gr.Textbox(
        show_label=False,
        placeholder="Enter text and press enter, or upload a pdf file",
    ).style(container=False)

    with gr.Row():
        with gr.Column(scale=0.70):
            btn = gr.UploadButton(
                "📁 Upload a pdf file",
                file_types=["file"],
            )
        with gr.Column(scale=0.30, min_width=0):
            clr = gr.Button(
                "Clear conversation",
            )

    txt.submit(
        add_text,
        inputs=[chat_box, txt],
        outputs=[chat_box, txt]
    ).then(bot, inputs=chat_box, outputs=chat_box)

    btn.upload(
        add_file,
        inputs=[chat_box, btn],
        outputs=[chat_box]
    ).then(bot, inputs=chat_box, outputs=chat_box)

    clr.click(
        clear_conversation,
        inputs=None,
        outputs=chat_box,
        queue=False
    )


app.launch()
