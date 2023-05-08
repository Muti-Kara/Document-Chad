import gradio as gr
import re

from reader import Reader
import chatai


chatbot = chatai.ChatBot(
    chat_model=chatai.CHAT_MODEL,
    embedding_model=chatai.EMBEDDING_MODEL,
    system_prompt=chatai.SYSTEM_PROMPT
)


def add_text(history, text):
    history = history + [(text, None)]
    return history, ""


def add_file(history, file):
    reader = Reader(file.name)
    chunks = reader.split_chunks(chatai.EMBEDDING_MODEL, max_chunk_size=512)
    history = history + [(str(reader), None)]
    return history


def bot(history):
    pattern = r"^File:\s\d+\ssentences in the document\.$"
    if re.match(pattern=pattern, string=history[-1][0]):
        response = "I received your file, now I must digest it :)"
    else:
        response = str(len(history))
    history[-1][1] = response
    return history


with gr.Blocks() as app:
    chatbot = gr.Chatbot().style(600)
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
        inputs=[chatbot, txt],
        outputs=[chatbot, txt]
    ).then(bot, inputs=chatbot, outputs=chatbot)

    btn.upload(
        add_file,
        inputs=[chatbot, btn],
        outputs=[chatbot]
    ).then(bot, inputs=chatbot, outputs=chatbot)

    clr.click(lambda: None, None, chatbot, queue=False)


app.launch()
