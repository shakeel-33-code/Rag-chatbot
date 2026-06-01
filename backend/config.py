import os
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

HF_API_KEY = os.getenv("HF_API_KEY", "")  # Set in .env file
EMBED_MODEL       = "BAAI/bge-large-en-v1.5"  # HF embedding model
LLM_MODEL         = "mistralai/Mistral-7B-Instruct-v0.2"  # HF LLM model

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
