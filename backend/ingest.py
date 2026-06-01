import io
import uuid
import pdfplumber
import chromadb
from chromadb.api.types import EmbeddingFunction, Embeddings
from huggingface_hub import InferenceClient
from chonkie import TokenChunker
import config


class InferenceClientEmbedding(EmbeddingFunction):
    """Embedding function backed by huggingface_hub.InferenceClient, which
    handles provider routing automatically and avoids raw DNS issues."""

    def __init__(self, api_key: str, model_name: str):
        self._client = InferenceClient(token=api_key)
        self._model_name = model_name

    def name(self) -> str:
        return f"hf-inference-client-{self._model_name}"

    def __call__(self, input: list) -> Embeddings:
        result = self._client.feature_extraction(input, model=self._model_name)
        return result.tolist()


# Module-level singletons
chroma_client = chromadb.PersistentClient(path="./chroma_db")
hf_ef = InferenceClientEmbedding(
    api_key=config.HF_API_KEY,
    model_name=config.EMBED_MODEL,
)
collection = chroma_client.get_or_create_collection(
    name="rag_docs",
    embedding_function=hf_ef,
)

def get_collection():
    return collection

def ingest_pdf(file_bytes: bytes, filename: str) -> int:
    # 1. Parse PDF
    text = ""
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
                
    # 2. Chunking
    chunker = TokenChunker(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )
    chunks = chunker.chunk(text)
    
    # 3. Add to ChromaDB
    if not chunks:
        return 0
        
    ids = [str(uuid.uuid4()) for _ in chunks]
    documents = [chunk.text for chunk in chunks]
    metadatas = [{"source": filename} for _ in chunks]
    
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    
    return len(chunks)
