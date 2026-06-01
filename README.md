<div align="center">

# 🤖 RAG Chatbot

**A production-ready Retrieval-Augmented Generation chatbot**  
*Upload any PDF → Ask questions → Get grounded, accurate answers*

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-1.1.1-FF6B35?style=for-the-badge&logo=databricks&logoColor=white)](https://www.trychroma.com)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Inference-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<br/>

<img src="https://raw.githubusercontent.com/shakeel-33-code/Rag-chatbot/main/assets/demo.png" alt="RAG Chatbot Demo" width="800"/>

</div>

---

## ✨ Features

| Feature | Details |
|---|---|
| 📄 **PDF Ingestion** | Upload any PDF and instantly index it |
| 🔍 **Semantic Search** | `BAAI/bge-large-en-v1.5` embeddings via HuggingFace |
| 🧠 **LLM Answers** | Powered by `Mistral-7B-Instruct-v0.2` |
| 💬 **Multi-turn Chat** | Full conversation history with context |
| ⚙️ **Live Settings** | Adjust temperature, tokens, top-k chunks — all from the UI |
| 🗄️ **Vector Store** | ChromaDB with persistent local storage |
| 🌐 **Zero-dependency Frontend** | Plain HTML/CSS/JS — no build step needed |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Browser (HTML/JS)                    │
│  Upload PDF ──► /upload      Ask Question ──► /chat     │
└────────────────────┬────────────────────────┬───────────┘
                     │                        │
┌────────────────────▼────────────────────────▼───────────┐
│                  FastAPI Backend                         │
│                                                          │
│   ingest.py          retriever.py          llm.py        │
│  ┌──────────┐       ┌────────────┐       ┌──────────┐   │
│  │ pdfplumb │       │  ChromaDB  │       │ Mistral  │   │
│  │  chonkie │──────►│  query()   │──────►│  7B via  │   │
│  │  embed   │       │  top-k     │       │   HF API │   │
│  └──────────┘       └────────────┘       └──────────┘   │
│        │                                                  │
│   ┌────▼──────────────────────────┐                      │
│   │  ChromaDB  (chroma_db/)       │                      │
│   │  BAAI/bge-large-en-v1.5       │                      │
│   └───────────────────────────────┘                      │
└──────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/shakeel-33-code/Rag-chatbot.git
cd Rag-chatbot
```

### 2. Create a virtual environment
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
pip install python-dotenv huggingface_hub
```

### 4. Configure your API key
```bash
cp .env.example .env
```
Open `.env` and set your [HuggingFace API token](https://huggingface.co/settings/tokens):
```env
HF_API_KEY=hf_your_token_here
```

### 5. Start the backend
```bash
cd backend
uvicorn main:app --reload
```

### 6. Open the frontend
Open `frontend/index.html` directly in your browser — no server needed.

---

## 📁 Project Structure

```
rag-project/
│
├── backend/
│   ├── main.py          # FastAPI app & API routes
│   ├── ingest.py        # PDF parsing, chunking & embedding
│   ├── retriever.py     # Semantic search via ChromaDB
│   ├── llm.py           # LLM prompt builder & inference
│   ├── config.py        # Central configuration
│   └── chroma_db/       # Auto-created vector store (gitignored)
│
├── frontend/
│   └── index.html       # Single-file UI (no build step)
│
├── .env                 # Your secrets (gitignored)
├── .env.example         # Template for .env
├── requirements.txt
└── README.md
```

---

## ⚙️ Configuration

All settings are adjustable **live from the UI** via the ⚙️ button, or by editing `backend/config.py`:

| Parameter | Default | Description |
|---|---|---|
| `EMBED_MODEL` | `BAAI/bge-large-en-v1.5` | Embedding model for indexing & retrieval |
| `LLM_MODEL` | `mistralai/Mistral-7B-Instruct-v0.2` | Language model for generating answers |
| `CHUNK_SIZE` | `400` | Tokens per chunk when splitting PDFs |
| `CHUNK_OVERLAP` | `50` | Overlap between consecutive chunks |
| `TOP_K` | `3` | Number of chunks retrieved per query |
| `MAX_NEW_TOKENS` | `512` | Max tokens the LLM can generate |
| `temperature` | `0.2` | LLM creativity (0 = deterministic, 2 = creative) |

---

## 🔌 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/upload` | Upload a PDF file for ingestion |
| `POST` | `/chat` | Send a question and get an answer |
| `GET` | `/settings` | Fetch current runtime settings |
| `POST` | `/settings` | Update settings without restarting |
| `GET` | `/health` | Health check |

### Example: Chat request
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is chunking?", "history": []}'
```

---

## 🛠️ Tech Stack

- **[FastAPI](https://fastapi.tiangolo.com/)** — High-performance Python API framework
- **[ChromaDB](https://www.trychroma.com/)** — Embedded vector database for semantic search
- **[chonkie](https://github.com/chonkie-ai/chonkie)** — Token-aware text chunking
- **[pdfplumber](https://github.com/jsvine/pdfplumber)** — Accurate PDF text extraction
- **[huggingface_hub](https://github.com/huggingface/huggingface_hub)** — Managed inference API client
- **[BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5)** — State-of-the-art embedding model
- **[Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)** — Instruction-tuned LLM

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">
  <sub>Built with ❤️ for the AI Builders Community</sub>
</div>
