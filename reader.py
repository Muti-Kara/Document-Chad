from nltk.tokenize import sent_tokenize
from pypdf import PdfReader
import tiktoken


class Reader:
    def __init__(self, file_name):
        reader = PdfReader(file_name)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + " "

        self.sentences = sent_tokenize(text)

    def __str__(self) -> str:
        return f"File: {len(self.sentences)} sentences in the document."

    def split_chunks(self, model_name: str, max_chunk_size: int) -> list[str]:
        encoding = tiktoken.encoding_for_model(model_name=model_name)
        encoded_sent_lens = [
            len(encoding.encode(sent)) for sent in self.sentences
        ]

        if len(encoded_sent_lens) == 0:
            raise Exception("Empty File!!!")

        text_chunks = []
        cur_token_cnt = encoded_sent_lens[0]
        chunk = [self.sentences[0]]
        for index in range(1, len(encoded_sent_lens)):
            if cur_token_cnt + encoded_sent_lens[index] < max_chunk_size:
                chunk.append(self.sentences[index])
                cur_token_cnt += encoded_sent_lens[index]
            else:
                text_chunks.append(chunk)
                chunk = [self.sentences[index]]
                cur_token_cnt = encoded_sent_lens[index]

        return ["\n".join(chunk) for chunk in text_chunks]
