import io
import os
import uuid

import chromadb
import pdfplumber
from chromadb.api.types import EmbeddingFunction, Embeddings
from chonkie import TokenChunker
from huggingface_hub import InferenceClient
from openai import OpenAI

import config
from observability import (
    record_exception,
    set_attribute,
    set_embedding_attributes,
    set_input,
    set_output,
    timed_span,
)
from retrieval.schemas import DocumentChunk
from retrieval.sparse import SpladeSparseEncoder
from storage.qdrant_store import QdrantVectorStore


class CompatibleEmbeddingFunction(EmbeddingFunction):
    """
    Prefer OpenAI-compatible embeddings when explicitly configured.
    Fall back to Hugging Face feature extraction when the router only supports chat.
    """

    def __init__(self, model_name: str):
        self._model_name = model_name
        self._openai_client = None
        self._openai_client_config = None
        self._hf_client = None
        self._hf_client_token = None

    def name(self) -> str:
        backend = "openai" if config.should_use_openai_embeddings() else "hf-feature-extraction"
        return f"{backend}-{self._model_name}"

    def __call__(self, input: list) -> Embeddings:
        if not input:
            return []

        if config.should_use_openai_embeddings():
            response = self._get_openai_client().embeddings.create(
                model=self._model_name,
                input=input,
                encoding_format="float",
            )
            return [item.embedding for item in response.data]

        with timed_span(
            "embedding",
            "embedding.duration_ms",
            {"embedding.input_count": len(input)},
            span_kind="EMBEDDING",
        ) as span:
            set_embedding_attributes(span, model_name=self._model_name, input_count=len(input))
            set_input(span, input, mime_type="application/json")
            try:
                result = self._get_hf_client().feature_extraction(input, model=self._model_name)
                embeddings = result.tolist() if hasattr(result, "tolist") else result
                dimension = _embedding_dimension(embeddings)
                set_embedding_attributes(
                    span,
                    model_name=self._model_name,
                    input_count=len(input),
                    dimension=dimension,
                )
                set_output(
                    span,
                    {"embedding_count": len(embeddings), "dimension": dimension},
                    mime_type="application/json",
                )
                return embeddings
            except Exception as exc:
                record_exception(span, exc)
                raise

    def _get_openai_client(self) -> OpenAI:
        api_key = config.get_embedding_openai_api_key()
        base_url = config.get_embedding_openai_base_url()
        if not api_key:
            raise RuntimeError(
                "EMBEDDING_OPENAI_API_KEY or OPENAI_API_KEY is not configured in .env."
            )

        current_config = (api_key, base_url)
        if self._openai_client is None or current_config != self._openai_client_config:
            self._openai_client = OpenAI(api_key=api_key, base_url=base_url)
            self._openai_client_config = current_config

        return self._openai_client

    def _get_hf_client(self) -> InferenceClient:
        token = config.get_openai_api_key()
        if not token:
            raise RuntimeError("OPENAI_API_KEY or HF_API_KEY is not configured in .env.")

        if self._hf_client is None or token != self._hf_client_token:
            self._hf_client = InferenceClient(token=token)
            self._hf_client_token = token

        return self._hf_client


CHROMA_DB_PATH = os.getenv(
    "CHROMA_DB_PATH",
    os.path.join(os.path.dirname(__file__), "chroma_db"),
)
CHROMA_RECOVERY_DB_PATH = os.getenv(
    "CHROMA_RECOVERY_DB_PATH",
    os.path.join(os.path.dirname(__file__), "chroma_db_recovered"),
)
embedding_function = CompatibleEmbeddingFunction(model_name=config.EMBED_MODEL)
_chroma_client = None
_collection = None
_active_chroma_db_path = None


def get_collection():
    global _chroma_client, _collection, _active_chroma_db_path
    if _collection is None:
        _chroma_client = _create_chroma_client()
        _collection = _chroma_client.get_or_create_collection(
            name=config.get_collection_name(),
            embedding_function=embedding_function,
        )
    return _collection


def get_active_chroma_db_path() -> str | None:
    return _active_chroma_db_path


def _create_chroma_client():
    global _active_chroma_db_path
    if os.path.exists(os.path.join(CHROMA_RECOVERY_DB_PATH, "chroma.sqlite3")):
        _active_chroma_db_path = CHROMA_RECOVERY_DB_PATH
        return chromadb.PersistentClient(path=CHROMA_RECOVERY_DB_PATH)

    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        _active_chroma_db_path = CHROMA_DB_PATH
        return client
    except BaseException:
        client = chromadb.PersistentClient(path=CHROMA_RECOVERY_DB_PATH)
        _active_chroma_db_path = CHROMA_RECOVERY_DB_PATH
        return client


def embed_texts(texts: list) -> Embeddings:
    return embedding_function(texts)


def ingest_pdf(
    file_bytes: bytes,
    filename: str,
    *,
    user_id: str | None = None,
    document_id: str | None = None,
) -> int:
    with timed_span(
        "pdf_parse",
        "pdf_parse.duration_ms",
        {"upload.filename": filename},
        span_kind="CHAIN",
    ) as span:
        page_texts = []
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            set_attribute(span, "pdf.page_count", len(pdf.pages))
            for page_number, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text()
                if page_text:
                    page_texts.append((page_number, page_text))
        text = "\n".join(page_text for _, page_text in page_texts)
        set_attribute(span, "pdf.extracted_chars", len(text))

    with timed_span(
        "chunking",
        "chunking.duration_ms",
        {
            "chunking.chunk_size": config.CHUNK_SIZE,
            "chunking.chunk_overlap": config.CHUNK_OVERLAP,
        },
        span_kind="CHAIN",
    ) as span:
        chunks = _chunk_pdf_pages(page_texts)
        set_attribute(span, "chunking.chunk_count", len(chunks))

    if not chunks:
        return 0

    ids = [str(uuid.uuid4()) for _ in chunks]
    documents = [chunk["text"] for chunk in chunks]
    metadatas = [{"source": filename} for _ in chunks]

    if config.use_qdrant():
        return _ingest_qdrant_chunks(
            ids=ids,
            documents=documents,
            chunks=chunks,
            filename=filename,
            user_id=user_id,
            document_id=document_id,
        )

    with timed_span(
        "vector_add",
        "vector_add.duration_ms",
        {
            "vector_store.vendor": "chromadb",
            "vector_store.collection": config.get_collection_name(),
            "vector_add.document_count": len(documents),
            "embedding.model_name": config.EMBED_MODEL,
            "embedding.provider": (
                "openai" if config.should_use_openai_embeddings() else "huggingface"
            ),
        },
        span_kind="RETRIEVER",
    ) as span:
        try:
            get_collection().add(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
            )
        except Exception as exc:
            record_exception(span, exc)
            raise

    return len(chunks)


def _chunk_pdf_pages(page_texts: list[tuple[int, str]]) -> list[dict]:
    chunker = TokenChunker(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    chunks = []
    for page_number, page_text in page_texts:
        for chunk_index_on_page, chunk in enumerate(chunker.chunk(page_text), start=1):
            text = chunk.text.strip()
            if not text:
                continue
            chunks.append(
                {
                    "text": text,
                    "page_number": page_number,
                    "chunk_index_on_page": chunk_index_on_page,
                }
            )
    return chunks


def _ingest_qdrant_chunks(
    *,
    ids: list[str],
    documents: list[str],
    chunks: list[dict],
    filename: str,
    user_id: str | None,
    document_id: str | None,
) -> int:
    effective_user_id = (user_id or config.DEFAULT_USER_ID).strip()
    if not effective_user_id:
        raise ValueError(
            "Qdrant ingestion requires user_id. Pass user_id or set DEFAULT_USER_ID."
        )

    effective_document_id = (document_id or str(uuid.uuid4())).strip()
    dense_vectors = embed_texts(documents)
    sparse_vectors = SpladeSparseEncoder().encode(documents)

    document_chunks = []
    for chunk_index, (chunk_id, document, chunk) in enumerate(
        zip(ids, documents, chunks),
        start=1,
    ):
        document_chunks.append(
            DocumentChunk(
                id=chunk_id,
                content=document,
                metadata={
                    "user_id": effective_user_id,
                    "document_id": effective_document_id,
                    "source": filename,
                    "file_type": "pdf",
                    "page_number": chunk["page_number"],
                    "chunk_index": chunk_index,
                    "chunk_index_on_page": chunk["chunk_index_on_page"],
                    "chunk_type": "child",
                    "parser": "pdfplumber",
                    "chunker": "token",
                },
            )
        )

    with timed_span(
        "vector_add",
        "vector_add.duration_ms",
        {
            "vector_store.vendor": "qdrant",
            "vector_store.collection": config.QDRANT_COLLECTION,
            "vector_add.document_count": len(document_chunks),
            "embedding.model_name": config.EMBED_MODEL,
            "sparse.model_name": config.QDRANT_SPARSE_MODEL,
        },
        span_kind="RETRIEVER",
    ) as span:
        try:
            QdrantVectorStore().upsert_chunks(
                document_chunks,
                dense_vectors,
                sparse_vectors,
            )
        except Exception as exc:
            record_exception(span, exc)
            raise

    return len(document_chunks)


def _embedding_dimension(embeddings: Embeddings) -> int:
    if not embeddings:
        return 0

    first_embedding = embeddings[0]
    if hasattr(first_embedding, "__len__"):
        return len(first_embedding)

    return 0
