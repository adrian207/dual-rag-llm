from fastapi import FastAPI, Request
from pydantic import BaseModel
from llama_index.core import VectorStoreIndex, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from chromadb import PersistentClient
import requests
import os

app = FastAPI()

# Global settings
Settings.embed_model = HuggingFaceEmbedding("all-MiniLM-L6-v2")

class Query(BaseModel):
    question: str
    file_ext: str = ""

def get_model(file_ext: str):
    if file_ext in ['.cs', '.ps1', '.yaml', '.yml', '.xaml']:
        return "qwen2.5-coder:32b-q4_K_M"
    else:
        return "deepseek-coder-v2:33b-q4_K_M"

def load_index(is_ms: bool):
    path = "/app/indexes/chroma_ms" if is_ms else "/app/indexes/chroma_oss"
    client = PersistentClient(path=path)
    coll = client.get_collection("msdocs" if is_ms else "ossdocs")
    vector_store = ChromaVectorStore(chroma_collection=coll)
    return VectorStoreIndex.from_vector_store(vector_store)

@app.post("/query")
def query(q: Query):
    is_ms = q.file_ext in ['.cs', '.ps1', '.yaml', '.yml', '.xaml']
    model = get_model(q.file_ext)

    # Auto-switch model in Ollama
    requests.post("http://ollama:11434/api/pull", json={"name": model})

    # Load correct index
    index = load_index(is_ms)
    retriever = index.as_retriever(similarity_top_k=3)
    nodes = retriever.retrieve(q.question)
    context = "\n".join([n.text for n in nodes])

    # Generate
    llm = Ollama(model=model, request_timeout=120)
    response = llm.complete(f"Context:\n{context}\n\nQuestion: {q.question}\nAnswer with code if needed.")
    
    return {"answer": response.text, "model": model, "source": "MS" if is_ms else "OSS"}