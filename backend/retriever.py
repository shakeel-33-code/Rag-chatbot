from ingest import embed_texts, get_collection
import config
from observability import (
    record_exception,
    set_attribute,
    set_input,
    set_output,
    set_retrieval_documents,
    timed_span,
)
from retrieval.schemas import RetrievedChunk
from retrieval.sparse import SpladeSparseEncoder
from storage.qdrant_store import QdrantVectorStore

def retrieve(query: str, top_k: int = config.TOP_K, user_id: str | None = None) -> str:
    query_embeddings = embed_texts([query])
    query_embedding = query_embeddings[0] if query_embeddings else []

    if config.use_qdrant():
        return _retrieve_from_qdrant(query, query_embedding, top_k=top_k, user_id=user_id)

    with timed_span(
        "vector_search",
        "vector_search.duration_ms",
        {
            "vector_store.vendor": "chromadb",
            "retrieval.top_k": top_k,
            "retrieval.max_context_tokens": config.MAX_CTX_TOKENS,
        },
        span_kind="RETRIEVER",
    ) as span:
        set_input(span, {"query": query, "top_k": top_k}, mime_type="application/json")
        try:
            collection = get_collection()
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            record_exception(span, e)
            raise

        chunks = results["documents"][0] if results["documents"] else []
        ids = results.get("ids", [[]])[0] if results.get("ids") else []
        distances = results.get("distances", [[]])[0] if results.get("distances") else []
        metadatas = results.get("metadatas", [[]])[0] if results.get("metadatas") else []

        context_chunks = []
        retrieval_documents = []
        current_tokens = 0
        context_budget_exhausted = False

        for idx, chunk in enumerate(chunks):
            # Fast approximation: len(text) // 4
            estimated_tokens = len(chunk) // 4
            included_in_context = (
                not context_budget_exhausted
                and current_tokens + estimated_tokens <= config.MAX_CTX_TOKENS
            )
            metadata = metadatas[idx] if idx < len(metadatas) and metadatas[idx] else {}
            distance = distances[idx] if idx < len(distances) else None

            retrieval_documents.append(
                {
                    "id": ids[idx] if idx < len(ids) else None,
                    "content": chunk,
                    "score": _distance_to_score(distance),
                    "metadata": {
                        **metadata,
                        "rank": idx + 1,
                        "distance": distance,
                        "chars": len(chunk),
                        "tokens_estimate": estimated_tokens,
                        "included_in_context": included_in_context,
                    },
                }
            )

            if not included_in_context:
                context_budget_exhausted = True
                continue
            context_chunks.append(chunk)
            current_tokens += estimated_tokens

        set_attribute(span, "retrieval.returned_chunks", len(context_chunks))
        set_attribute(span, "retrieval.candidate_chunks", len(chunks))
        set_attribute(span, "retrieval.context_tokens_estimate", current_tokens)
        set_retrieval_documents(span, retrieval_documents)

        context = "\n\n---\n\n".join(context_chunks)
        set_output(
            span,
            {
                "context": context,
                "returned_chunks": len(context_chunks),
                "candidate_chunks": len(chunks),
                "context_tokens_estimate": current_tokens,
            },
            mime_type="application/json",
        )

        return context


def _retrieve_from_qdrant(
    query: str,
    query_embedding: list[float],
    *,
    top_k: int,
    user_id: str | None,
) -> str:
    effective_user_id = (user_id or config.DEFAULT_USER_ID).strip()
    if not effective_user_id:
        raise ValueError(
            "Qdrant retrieval requires user_id. Pass user_id or set DEFAULT_USER_ID."
        )

    with timed_span(
        "hybrid_search",
        "hybrid_search.duration_ms",
        {
            "vector_store.vendor": "qdrant",
            "vector_store.collection": config.QDRANT_COLLECTION,
            "retrieval.top_k": top_k,
            "retrieval.max_context_tokens": config.MAX_CTX_TOKENS,
        },
        span_kind="RETRIEVER",
    ) as span:
        set_input(span, {"query": query, "top_k": top_k}, mime_type="application/json")
        try:
            query_sparse_vector = SpladeSparseEncoder().encode([query])[0]
            chunks = QdrantVectorStore().hybrid_search(
                query_embedding,
                query_sparse_vector,
                user_id=effective_user_id,
                top_k=top_k,
            )
        except Exception as e:
            record_exception(span, e)
            raise

        context, retrieval_documents, current_tokens = _build_context(chunks)
        set_attribute(span, "retrieval.returned_chunks", len(context.split("\n\n---\n\n")) if context else 0)
        set_attribute(span, "retrieval.candidate_chunks", len(chunks))
        set_attribute(span, "retrieval.context_tokens_estimate", current_tokens)
        set_retrieval_documents(span, retrieval_documents)
        set_output(
            span,
            {
                "context": context,
                "returned_chunks": len(retrieval_documents),
                "candidate_chunks": len(chunks),
                "context_tokens_estimate": current_tokens,
            },
            mime_type="application/json",
        )
        return context


def _build_context(chunks: list[RetrievedChunk]) -> tuple[str, list[dict], int]:
    context_chunks = []
    retrieval_documents = []
    current_tokens = 0
    context_budget_exhausted = False

    for idx, chunk in enumerate(chunks):
        estimated_tokens = len(chunk.content) // 4
        included_in_context = (
            not context_budget_exhausted
            and current_tokens + estimated_tokens <= config.MAX_CTX_TOKENS
        )
        retrieval_documents.append(
            {
                "id": chunk.id,
                "content": chunk.content,
                "score": chunk.score,
                "metadata": {
                    **chunk.metadata,
                    "rank": idx + 1,
                    "dense_score": chunk.dense_score,
                    "sparse_score": chunk.sparse_score,
                    "fusion_score": chunk.fusion_score,
                    "chars": len(chunk.content),
                    "tokens_estimate": estimated_tokens,
                    "included_in_context": included_in_context,
                },
            }
        )

        if not included_in_context:
            context_budget_exhausted = True
            continue
        context_chunks.append(chunk.content)
        current_tokens += estimated_tokens

    return "\n\n---\n\n".join(context_chunks), retrieval_documents, current_tokens


def _distance_to_score(distance):
    if distance is None:
        return None

    try:
        return round(1 / (1 + float(distance)), 6)
    except Exception:
        return None
