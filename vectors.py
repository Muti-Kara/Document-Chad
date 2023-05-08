from dotenv import load_dotenv
import pinecone
import os

load_dotenv()

pinecone.init(
    api_key=os.getenv("PINECONE_API"),
    environment=os.getenv("PINECONE_ENV")
)


def get_index(index_name: str, dims: int) -> pinecone.Index:
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(index_name, dimension=dims)

    index = pinecone.Index(index_name)

    if index.describe_index_stats()["dimension"] != dims:
        pinecone.delete_index(index_name)
        pinecone.create_index(index_name, dimension=dims)
        index = pinecone.Index(index_name)

    return index
