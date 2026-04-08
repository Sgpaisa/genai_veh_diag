from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.embeddings import BaseEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from vertexai.language_models import TextEmbeddingModel
from google.cloud import aiplatform
from dotenv import load_dotenv
from typing import List
import os

load_dotenv()

PROJECT_ID = "vehicle-diagnostics-491610"
REGION     = "asia-south1"

aiplatform.init(project=PROJECT_ID, location=REGION)

# LLM — uses api_key (works fine for generation)
Settings.llm = GoogleGenAI(
    model="gemini-2.5-flash",
    api_key=os.getenv("GEMINI_API_KEY")
)


# Custom embedding wrapper — uses vertexai TextEmbeddingModel
# Same model that already works in etl.py
class VertexEmbedding(BaseEmbedding):
    def __init__(self):
        super().__init__()
        self._model = TextEmbeddingModel.from_pretrained("text-embedding-004")

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._model.get_embeddings([text])[0].values

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._model.get_embeddings([query])[0].values

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return [e.values for e in self._model.get_embeddings(texts)]


Settings.embed_model = VertexEmbedding()

# Load OBD-II manual documents
print("Loading documents...")
documents = SimpleDirectoryReader("./obd_docs").load_data()
print(f"Loaded {len(documents)} document(s)")

# Build vector index
print("Building index...")
index = VectorStoreIndex.from_documents(documents)
print("Index built successfully")

query_engine = index.as_query_engine(similarity_top_k=3)


def ask(question: str):
    print(f"\nQ: {question}")
    response = query_engine.query(question)
    print(f"A: {response}")
    if response.source_nodes:
        print(f"Source: {response.source_nodes[0].metadata}")


if __name__ == "__main__":
    ask("Engine stalling at idle with misfire codes — what should I check?")
    ask("What is the severity of P0171 and what causes it?")
    ask("How do I repair a thermostat-related fault code?")
