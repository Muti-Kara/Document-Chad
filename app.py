import gradio as gr
import tiktoken
import backoff

from chatai import ChatBot, Embedder
from vectors import VectorManager
from reader import Reader


SRC_CHUNK_COUNT = 8
MAX_USER_TOKEN = 512
MAX_ANSWER_TOKEN = 512

encoder = tiktoken.encoding_for_model(model_name="text-embedding-ada-002")
chat_bot = ChatBot(model="gpt-3.5-turbo", answer_token=512)
embedder = Embedder(model="text-embedding-ada-002")
vector_manager = VectorManager(index_name="qa-embedding", dims=1536, namespace="namespace")
reader = Reader(encoder=encoder, max_chunk_size=128, batch_len=50)


def add_text(history, text):
    history = history + [(text, None)]
    return history, ""


def clear_conversation():
    vector_manager.clr()
    chat_bot.clear_all()
    return None


def add_file(file):
    pages = reader.read_file(file.name)
    sentences = reader.separate_sentences(" ".join(pages))
    processed_sents = [sent.replace(" ", "") for sent in sentences]
    chunks = reader.split_chunks(processed_sents)
    batches = reader.batch_chunks(chunks)

    embeddings = []
    for batch in batches:
        embedding_batch = []
        for chunk in batch:
            try:
                embedding_batch.append(embedder.get_embedding(chunk))
            except backoff.MaxRetriesExceeded:
                print("Exceeded maximum attempts.")
            except Exception as e:
                print(f"Error occurred {str(e)}")
        embeddings.append(embedding_batch)

    for index in range(len(batches)):
        vector_manager.put(
            chunks=batches[index],
            embeddings=embeddings[index]
        )

    return tuple(["File has uploaded"] * SRC_CHUNK_COUNT)


def bot(history):
    user_prompt = history[-1][0]
    srcs = ["not found"] * SRC_CHUNK_COUNT

    if len(encoder.encode(user_prompt)) > MAX_USER_TOKEN:
        response = "This is much longer message than I can answer :("
    else:
        embedding = embedder.get_embedding(user_prompt)
        srcs = vector_manager.get(embedding=embedding, top_k=SRC_CHUNK_COUNT) + srcs
        chat_bot.add_question(user_prompt, srcs)
        response = chat_bot.get_response()

    srcs = [src[:40] for src in srcs]
    history[-1][1] = response
    return history, *srcs


with gr.Blocks() as app:
    with gr.Row().style(equal_height=True):
        with gr.Column(scale=0.8):
            chat_box = gr.Chatbot().style(600)
            txt = gr.Textbox(
                show_label=False,
                placeholder="Enter text and press enter, or upload a pdf file",
            ).style(container=False)
        with gr.Column(scale=0.2):
            srcs = [gr.Textbox(label=f"Source {i+1}") for i in range(SRC_CHUNK_COUNT)]

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
    ).then(bot, inputs=chat_box, outputs=[chat_box, *srcs])

    btn.upload(
        add_file,
        inputs=[btn],
        outputs=[*srcs]
    )

    clr.click(
        clear_conversation,
        inputs=None,
        outputs=chat_box,
        queue=False
    )


app.launch()
