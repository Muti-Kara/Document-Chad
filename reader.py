from nltk.tokenize import sent_tokenize
from PyPDF2 import PdfReader


class Reader:
    def __init__(self, encoder, max_chunk_size: int, batch_len: int):
        self.max_chunk_size = max_chunk_size
        self.batch_len = batch_len
        self.encoder = encoder

    def read_file(self, file_name: str) -> list[str]:
        return [page.extract_text() for page in PdfReader(file_name).pages]

    def separate_sentences(self, text: str) -> list[str]:
        return sent_tokenize(text)

    def split_chunks(self, sentences: list[str]) -> list[str]:
        sent_lens = [len(self.encoder.encode(sent)) for sent in sentences]
        cur_token_cnt = 0
        chunk = []
        chunks = []

        for index, sent in enumerate(sentences):
            sent_len = sent_lens[index]
            if cur_token_cnt + sent_len <= self.max_chunk_size:
                chunk.append(sent)
                cur_token_cnt += sent_len
            else:
                chunks.append(chunk)
                chunk = [sent]
                cur_token_cnt = sent_len
        if chunk:
            chunks.append(chunk)
        return [" ".join(chunk) for chunk in chunks]

    def batch_chunks(self, chunks: list[str]) -> list[list[str]]:
        return [chunks[i:i + self.batch_len] for i in range(0, len(chunks), self.batch_len)]
