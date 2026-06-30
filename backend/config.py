import os
from dotenv import dotenv_values, load_dotenv

ENV_PATH = os.path.join(os.path.dirname(__file__), "..", ".env")

load_dotenv(dotenv_path=ENV_PATH, override=True)

DEFAULT_OPENAI_BASE_URL = "https://router.huggingface.co/v1"
DEFAULT_EMBEDDING_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_OLLAMA_BASE_URL = "https://ollama.com/v1"

VECTOR_DB = os.getenv("VECTOR_DB", "chroma").strip().lower()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "huggingface").strip().lower()

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-large-en-v1.5")
LLM_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
LLM_FALLBACK_MODELS = [
    model.strip()
    for model in os.getenv(
        "LLM_FALLBACK_MODELS",
        "mistralai/Mistral-7B-Instruct-v0.3,HuggingFaceH4/zephyr-7b-beta",
    ).split(",")
    if model.strip()
]

PHOENIX_TRACING_ENABLED = os.getenv("PHOENIX_TRACING_ENABLED", "true").lower() == "true"
PHOENIX_PROJECT_NAME = os.getenv("PHOENIX_PROJECT_NAME", "rag-project")
PHOENIX_COLLECTOR_ENDPOINT = os.getenv(
    "PHOENIX_COLLECTOR_ENDPOINT",
    "http://localhost:6006/v1/traces",
)
PHOENIX_CAPTURE_CONTENT = os.getenv("PHOENIX_CAPTURE_CONTENT", "true").lower() == "true"
PHOENIX_CAPTURE_CONTENT_MAX_CHARS = int(os.getenv("PHOENIX_CAPTURE_CONTENT_MAX_CHARS", "4000"))

CHUNK_SIZE        = 400     # tokens per chunk
CHUNK_OVERLAP     = 50      # overlap between chunks

TOP_K             = 3       # number of chunks to retrieve
MAX_CTX_TOKENS    = 1800    # hard ceiling for context fed to LLM
MAX_NEW_TOKENS    = 512     # max tokens for LLM to generate

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333").strip()
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "").strip() or None
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "rag_chunks").strip()
QDRANT_DENSE_VECTOR_NAME = os.getenv("QDRANT_DENSE_VECTOR_NAME", "dense").strip()
QDRANT_SPARSE_VECTOR_NAME = os.getenv("QDRANT_SPARSE_VECTOR_NAME", "sparse").strip()
QDRANT_DISTANCE = os.getenv("QDRANT_DISTANCE", "Cosine").strip()
QDRANT_VECTOR_SIZE = int(os.getenv("QDRANT_VECTOR_SIZE", "1024"))
QDRANT_SPARSE_MODEL = os.getenv("QDRANT_SPARSE_MODEL", "").strip()
QDRANT_HYBRID_RRF_K = int(os.getenv("QDRANT_HYBRID_RRF_K", "60"))
QDRANT_DENSE_PREFETCH_LIMIT = int(os.getenv("QDRANT_DENSE_PREFETCH_LIMIT", "20"))
QDRANT_SPARSE_PREFETCH_LIMIT = int(os.getenv("QDRANT_SPARSE_PREFETCH_LIMIT", "20"))
DEFAULT_USER_ID = os.getenv("DEFAULT_USER_ID", "").strip()

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful and friendly assistant. "
    "For greetings, casual conversation, or questions unrelated to any document, respond naturally and conversationally. "
    "When the user asks about a document or a topic covered in the context below, answer using ONLY that context. "
    "If a document-related question cannot be answered from the context, say so clearly."
)


def _env_values() -> dict:
    return dotenv_values(ENV_PATH)


def get_openai_base_url() -> str:
    env_values = _env_values()
    return (
        env_values.get("OPENAI_BASE_URL")
        or os.getenv("OPENAI_BASE_URL", "")
        or DEFAULT_OPENAI_BASE_URL
    )


def get_llm_provider() -> str:
    env_values = _env_values()
    provider = (
        env_values.get("LLM_PROVIDER")
        or os.getenv("LLM_PROVIDER", "")
        or LLM_PROVIDER
    ).strip().lower()

    aliases = {
        "hf": "huggingface",
        "hugging_face": "huggingface",
        "huggingface": "huggingface",
        "nvidia": "nvidia",
        "nvidea": "nvidia",
        "nv": "nvidia",
        "ollama": "ollama",
    }
    normalized = aliases.get(provider)
    if not normalized:
        raise ValueError(
            "Unsupported LLM_PROVIDER. Use one of: huggingface, nvidia, ollama."
        )
    return normalized


def get_llm_base_url() -> str:
    env_values = _env_values()
    provider = get_llm_provider()
    if provider == "huggingface":
        return (
            env_values.get("HUGGINGFACE_BASE_URL")
            or os.getenv("HUGGINGFACE_BASE_URL", "")
            or env_values.get("OPENAI_BASE_URL")
            or os.getenv("OPENAI_BASE_URL", "")
            or DEFAULT_OPENAI_BASE_URL
        )
    if provider == "nvidia":
        return (
            env_values.get("NVIDIA_BASE_URL")
            or os.getenv("NVIDIA_BASE_URL", "")
            or DEFAULT_NVIDIA_BASE_URL
        )
    return (
        env_values.get("OLLAMA_BASE_URL")
        or os.getenv("OLLAMA_BASE_URL", "")
        or DEFAULT_OLLAMA_BASE_URL
    )


def get_openai_api_key() -> str:
    """Read the current chat provider key from .env without requiring a restart."""
    env_values = _env_values()
    return (
        env_values.get("OPENAI_API_KEY")
        or os.getenv("OPENAI_API_KEY", "")
        or env_values.get("HF_API_KEY")
        or os.getenv("HF_API_KEY", "")
        or env_values.get("HF_TOKEN")
        or os.getenv("HF_TOKEN", "")
    )


def get_llm_api_key() -> str:
    env_values = _env_values()
    provider = get_llm_provider()
    if provider == "huggingface":
        return (
            env_values.get("HUGGINGFACE_API_KEY")
            or os.getenv("HUGGINGFACE_API_KEY", "")
            or get_openai_api_key()
        )
    if provider == "nvidia":
        return (
            env_values.get("NVIDIA_API_KEY")
            or os.getenv("NVIDIA_API_KEY", "")
        )
    return (
        env_values.get("OLLAMA_API_KEY")
        or os.getenv("OLLAMA_API_KEY", "")
    )


def get_llm_model() -> str:
    env_values = _env_values()
    provider = get_llm_provider()
    provider_defaults = {
        "huggingface": "Qwen/Qwen2.5-7B-Instruct",
        "nvidia": "meta/llama-3.1-8b-instruct",
        "ollama": "gpt-oss:20b",
    }
    provider_specific_keys = {
        "huggingface": "HUGGINGFACE_LLM_MODEL",
        "nvidia": "NVIDIA_LLM_MODEL",
        "ollama": "OLLAMA_LLM_MODEL",
    }
    key = provider_specific_keys[provider]
    return (
        env_values.get(key)
        or os.getenv(key, "")
        or env_values.get("LLM_MODEL")
        or os.getenv("LLM_MODEL", "")
        or provider_defaults[provider]
    )


def get_llm_fallback_models() -> list[str]:
    env_values = _env_values()
    provider = get_llm_provider()
    provider_specific_keys = {
        "huggingface": "HUGGINGFACE_LLM_FALLBACK_MODELS",
        "nvidia": "NVIDIA_LLM_FALLBACK_MODELS",
        "ollama": "OLLAMA_LLM_FALLBACK_MODELS",
    }
    raw_value = (
        env_values.get(provider_specific_keys[provider])
        or os.getenv(provider_specific_keys[provider], "")
        or env_values.get("LLM_FALLBACK_MODELS")
        or os.getenv("LLM_FALLBACK_MODELS", "")
    )
    return [model.strip() for model in raw_value.split(",") if model.strip()]


def get_embedding_openai_base_url() -> str:
    env_values = _env_values()
    return (
        env_values.get("EMBEDDING_OPENAI_BASE_URL")
        or os.getenv("EMBEDDING_OPENAI_BASE_URL", "")
        or env_values.get("OPENAI_BASE_URL")
        or os.getenv("OPENAI_BASE_URL", "")
        or DEFAULT_EMBEDDING_OPENAI_BASE_URL
    )


def get_embedding_openai_api_key() -> str:
    """Read the current embedding provider key from .env without requiring a restart."""
    env_values = _env_values()
    return (
        env_values.get("EMBEDDING_OPENAI_API_KEY")
        or os.getenv("EMBEDDING_OPENAI_API_KEY", "")
        or env_values.get("OPENAI_API_KEY")
        or os.getenv("OPENAI_API_KEY", "")
        or env_values.get("HF_API_KEY")
        or os.getenv("HF_API_KEY", "")
        or env_values.get("HF_TOKEN")
        or os.getenv("HF_TOKEN", "")
    )


def should_use_openai_embeddings() -> bool:
    env_values = _env_values()
    explicit_embedding_key = env_values.get("EMBEDDING_OPENAI_API_KEY") or os.getenv(
        "EMBEDDING_OPENAI_API_KEY", ""
    )
    explicit_embedding_base_url = env_values.get("EMBEDDING_OPENAI_BASE_URL") or os.getenv(
        "EMBEDDING_OPENAI_BASE_URL", ""
    )
    explicit_openai_key = env_values.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY", "")
    explicit_openai_base_url = env_values.get("OPENAI_BASE_URL") or os.getenv("OPENAI_BASE_URL", "")
    hf_key = (
        env_values.get("HF_API_KEY")
        or os.getenv("HF_API_KEY", "")
        or env_values.get("HF_TOKEN")
        or os.getenv("HF_TOKEN", "")
    )

    if explicit_embedding_key or explicit_embedding_base_url:
        return True

    if explicit_openai_key and not hf_key:
        return True

    return bool(
        explicit_openai_key
        and explicit_openai_base_url
        and explicit_openai_base_url.rstrip("/") != DEFAULT_OPENAI_BASE_URL
    )


def get_collection_name() -> str:
    configured_name = os.getenv("CHROMA_COLLECTION_NAME", "").strip()
    if configured_name:
        return configured_name

    normalized_model = "".join(
        char.lower() if char.isalnum() else "_"
        for char in EMBED_MODEL
    ).strip("_")
    return f"rag_docs_{normalized_model or 'default'}"


def use_qdrant() -> bool:
    return VECTOR_DB == "qdrant"
