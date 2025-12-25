import os
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings


VECTOR_DB_DIR = "vector_db"
DATA_DIR = "data/hr_knowledge"


def build_vector_store():
    texts = []

    for file in os.listdir(DATA_DIR):
        if file.endswith(".txt"):
            with open(os.path.join(DATA_DIR, file), "r", encoding="utf-8") as f:
                texts.append(f.read())

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents(texts)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vectordb = Chroma.from_documents(
        docs,
        embedding=embeddings,
        persist_directory=VECTOR_DB_DIR
    )

    vectordb.persist()
    return vectordb


def load_vector_store():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma(
        persist_directory=VECTOR_DB_DIR,
        embedding_function=embeddings
    )
