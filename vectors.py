from dotenv import load_dotenv
import pinecone
import os

load_dotenv()

pinecone.init(
    api_key=os.getenv("PINECONE_API"),
    environment=os.getenv("PINECONE_ENV")
)


class VectorManager:
    def __init__(self, index_name: str, namespace: str, dims: int):
        self.pinecone_index = pinecone.Index(index_name)
        if self.pinecone_index.describe_index_stats()["dimension"] != dims:
            raise Exception("Index dimension is incompatible with embedding dimension")
        self.namespace = namespace
        self.dims = dims
        self.hash2string = dict()
        self.clr()

    def put(self, chunks: list[str], embeddings: list[list[float]]) -> bool:
        if len(chunks) != len(embeddings):
            raise Exception("invalid chunk and embedding list")
        if len(embeddings) > 0 and len(embeddings[0]) != self.dims:
            raise Exception("invalid embedding dimension")
        hashed_chunks = list(map(str, map(hash, chunks)))
        self.hash2string = dict(zip(hashed_chunks, chunks))
        return self.pinecone_index.upsert(
            vectors=list(zip(hashed_chunks, embeddings)),
            namespace=self.namespace
        )["upserted_count"] == len(chunks)

    def get(self, embedding: list[float], top_k: int) -> list[str]:
        def match2string(hashed_result: dict) -> str:
            return self.hash2string.get(hashed_result["id"], None)
        return list(map(
            match2string,
            self.pinecone_index.query(
                vector=embedding,
                top_k=top_k,
                namespace=self.namespace
            )["matches"]
        ))

    def clr(self):
        self.pinecone_index.delete(deleteAll=True, namespace=self.namespace)
