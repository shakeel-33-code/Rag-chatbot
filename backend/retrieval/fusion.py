from retrieval.schemas import RetrievedChunk


def reciprocal_rank_fusion(
    dense_results: list[RetrievedChunk],
    sparse_results: list[RetrievedChunk],
    *,
    k: int = 60,
    limit: int,
) -> list[RetrievedChunk]:
    by_id: dict[str, RetrievedChunk] = {}
    scores: dict[str, float] = {}

    for rank, chunk in enumerate(dense_results, start=1):
        if not chunk.id:
            continue
        by_id[chunk.id] = chunk
        scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (k + rank)

    for rank, chunk in enumerate(sparse_results, start=1):
        if not chunk.id:
            continue
        existing = by_id.get(chunk.id)
        if existing is None:
            by_id[chunk.id] = chunk
        else:
            by_id[chunk.id] = RetrievedChunk(
                id=existing.id,
                content=existing.content,
                score=existing.score,
                dense_score=existing.dense_score,
                sparse_score=chunk.sparse_score,
                fusion_score=existing.fusion_score,
                metadata={**existing.metadata, **chunk.metadata},
            )
        scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (k + rank)

    ranked_ids = sorted(scores, key=lambda chunk_id: scores[chunk_id], reverse=True)
    fused = []
    for chunk_id in ranked_ids[:limit]:
        chunk = by_id[chunk_id]
        fused.append(
            RetrievedChunk(
                id=chunk.id,
                content=chunk.content,
                score=round(scores[chunk_id], 6),
                dense_score=chunk.dense_score,
                sparse_score=chunk.sparse_score,
                fusion_score=round(scores[chunk_id], 6),
                metadata=chunk.metadata,
            )
        )
    return fused
