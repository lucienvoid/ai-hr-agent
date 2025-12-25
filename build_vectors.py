from dotenv import load_dotenv
load_dotenv()

from core.vector_store import build_vector_store

build_vector_store()
print("Vector DB built successfully")
