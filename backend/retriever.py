from ingest import get_collection
import config

def retrieve(query: str, top_k: int = config.TOP_K) -> str:
    collection = get_collection()
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )
    
    chunks = results["documents"][0] if results["documents"] else []
    
    context_chunks = []
    current_tokens = 0
    
    for chunk in chunks:
        # Fast approximation: len(text) // 4
        estimated_tokens = len(chunk) // 4
        if current_tokens + estimated_tokens > config.MAX_CTX_TOKENS:
            break
        context_chunks.append(chunk)
        current_tokens += estimated_tokens
        
    return "\n\n---\n\n".join(context_chunks)
