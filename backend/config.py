import os
from dotenv import dotenv_values, load_dotenv

ENV_PATH = os.path.join(os.path.dirname(__file__), "..", ".env")

load_dotenv(dotenv_path=ENV_PATH, override=True)

DEFAULT_OPENAI_BASE_URL = "https://router.huggingface.co/v1"
DEFAULT_EMBEDDING_OPENAI_BASE_URL = "https://api.openai.com/v1"

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
