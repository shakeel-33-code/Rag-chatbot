from typing import Any

import config
from retrieval.fusion import reciprocal_rank_fusion
from retrieval.schemas import DocumentChunk, RetrievedChunk
from retrieval.sparse import SparseVector


class QdrantDependencyError(RuntimeError):
    pass


class QdrantMetadataError(ValueError):
    pass


REQUIRED_PAYLOAD_FIELDS = {
    "user_id",
    "document_id",
    "chunk_id",
    "source",
    "file_type",
    "page_number",
    "content",
    "dense_model",
    "sparse_model",
}


class QdrantVectorStore:
    def __init__(self) -> None:
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:
            raise QdrantDependencyError(
                "qdrant-client is not installed. Add qdrant-client to requirements "
                "and install dependencies before setting VECTOR_DB=qdrant."
            ) from exc

        self._client = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)

    def ensure_collection(self) -> None:
        try:
            from qdrant_client import models
        except ImportError as exc:
            raise QdrantDependencyError("qdrant-client is not installed.") from exc

        existing = {collection.name for collection in self._client.get_collections().collections}
        if config.QDRANT_COLLECTION in existing:
            self._ensure_payload_indexes(models)
            return

        self._client.create_collection(
            collection_name=config.QDRANT_COLLECTION,
            vectors_config={
                config.QDRANT_DENSE_VECTOR_NAME: models.VectorParams(
                    size=config.QDRANT_VECTOR_SIZE,
                    distance=_qdrant_distance(models, config.QDRANT_DISTANCE),
                )
            },
            sparse_vectors_config={
                config.QDRANT_SPARSE_VECTOR_NAME: models.SparseVectorParams()
            },
        )
        self._ensure_payload_indexes(models)

    def _ensure_payload_indexes(self, models: Any) -> None:
        self._client.create_payload_index(
            collection_name=config.QDRANT_COLLECTION,
            field_name="user_id",
            field_schema=models.PayloadSchemaType.KEYWORD,
            wait=True,
        )

    def upsert_chunks(
        self,
        chunks: list[DocumentChunk],
        dense_vectors: list[list[float]],
        sparse_vectors: list[SparseVector],
    ) -> None:
        if not (len(chunks) == len(dense_vectors) == len(sparse_vectors)):
            raise ValueError("Chunks, dense vectors, and sparse vectors must have the same length.")

        try:
            from qdrant_client import models
        except ImportError as exc:
            raise QdrantDependencyError("qdrant-client is not installed.") from exc

        self.ensure_collection()
        points = []
        for chunk, dense_vector, sparse_vector in zip(chunks, dense_vectors, sparse_vectors):
            payload = _build_payload(chunk)
            points.append(
                models.PointStruct(
                    id=chunk.id,
                    vector={
                        config.QDRANT_DENSE_VECTOR_NAME: dense_vector,
                        config.QDRANT_SPARSE_VECTOR_NAME: models.SparseVector(
                            indices=sparse_vector.indices,
                            values=sparse_vector.values,
                        ),
                    },
                    payload=payload,
                )
            )

        self._client.upsert(collection_name=config.QDRANT_COLLECTION, points=points)

    def dense_search(
        self,
        query_vector: list[float],
        *,
        user_id: str,
        top_k: int,
    ) -> list[RetrievedChunk]:
        self.ensure_collection()
        response = self._client.query_points(
            collection_name=config.QDRANT_COLLECTION,
            query=query_vector,
            using=config.QDRANT_DENSE_VECTOR_NAME,
            query_filter=_user_filter(user_id),
            limit=top_k,
            with_payload=True,
        )
        return [_scored_point_to_chunk(point, "dense") for point in response.points]

    def sparse_search(
        self,
        query_sparse_vector: SparseVector,
        *,
        user_id: str,
        top_k: int,
    ) -> list[RetrievedChunk]:
        try:
            from qdrant_client import models
        except ImportError as exc:
            raise QdrantDependencyError("qdrant-client is not installed.") from exc

        self.ensure_collection()
        response = self._client.query_points(
            collection_name=config.QDRANT_COLLECTION,
            query=models.SparseVector(
                indices=query_sparse_vector.indices,
                values=query_sparse_vector.values,
            ),
            using=config.QDRANT_SPARSE_VECTOR_NAME,
            query_filter=_user_filter(user_id),
            limit=top_k,
            with_payload=True,
        )
        return [_scored_point_to_chunk(point, "sparse") for point in response.points]

    def hybrid_search(
        self,
        query_vector: list[float],
        query_sparse_vector: SparseVector,
        *,
        user_id: str,
        top_k: int,
    ) -> list[RetrievedChunk]:
        dense_results = self.dense_search(
            query_vector,
            user_id=user_id,
            top_k=max(top_k, config.QDRANT_DENSE_PREFETCH_LIMIT),
        )
        sparse_results = self.sparse_search(
            query_sparse_vector,
            user_id=user_id,
            top_k=max(top_k, config.QDRANT_SPARSE_PREFETCH_LIMIT),
        )
        return reciprocal_rank_fusion(
            dense_results,
            sparse_results,
            k=config.QDRANT_HYBRID_RRF_K,
            limit=top_k,
        )


def _build_payload(chunk: DocumentChunk) -> dict[str, Any]:
    payload = {
        **chunk.metadata,
        "chunk_id": chunk.id,
        "content": chunk.content,
        "dense_model": config.EMBED_MODEL,
        "sparse_model": config.QDRANT_SPARSE_MODEL,
    }
    missing = sorted(field for field in REQUIRED_PAYLOAD_FIELDS if not payload.get(field))
    if missing:
        raise QdrantMetadataError(
            "Qdrant ingestion requires complete metadata. Missing fields: "
            + ", ".join(missing)
        )
    return payload


def _user_filter(user_id: str):
    if not user_id:
        raise QdrantMetadataError("Qdrant retrieval requires a user_id payload filter.")

    try:
        from qdrant_client import models
    except ImportError as exc:
        raise QdrantDependencyError("qdrant-client is not installed.") from exc

    return models.Filter(
        must=[
            models.FieldCondition(
                key="user_id",
                match=models.MatchValue(value=user_id),
            )
        ]
    )


def _scored_point_to_chunk(point: Any, score_type: str) -> RetrievedChunk:
    payload = dict(point.payload or {})
    content = str(payload.get("content", ""))
    metadata = {key: value for key, value in payload.items() if key != "content"}
    score = float(point.score) if point.score is not None else None
    return RetrievedChunk(
        id=str(point.id) if point.id is not None else None,
        content=content,
        score=score,
        dense_score=score if score_type == "dense" else None,
        sparse_score=score if score_type == "sparse" else None,
        metadata=metadata,
    )


def _qdrant_distance(models: Any, distance_name: str):
    normalized = distance_name.strip().upper()
    distances = {
        "COSINE": models.Distance.COSINE,
        "DOT": models.Distance.DOT,
        "EUCLID": models.Distance.EUCLID,
        "MANHATTAN": models.Distance.MANHATTAN,
    }
    if normalized not in distances:
        raise ValueError(f"Unsupported Qdrant distance: {distance_name}")
    return distances[normalized]
