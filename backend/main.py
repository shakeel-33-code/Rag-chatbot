from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import config
from ingest import ingest_pdf
from retriever import retrieve
from llm import build_prompt, ask_llm

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

settings = {
    "system_prompt": config.DEFAULT_SYSTEM_PROMPT,
    "temperature": 0.2,
    "max_new_tokens": config.MAX_NEW_TOKENS,
    "top_k": config.TOP_K,
}

class ChatRequest(BaseModel):
    question: str
    history: List[Dict[str, str]] = []

class SettingsRequest(BaseModel):
    system_prompt: str
    temperature: float
    max_new_tokens: int
    top_k: int

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    file_bytes = await file.read()
    chunks_ingested = ingest_pdf(file_bytes, file.filename)
    return {"message": f"Ingested {chunks_ingested} chunks from {file.filename}"}

@app.post("/chat")
def chat(request: ChatRequest):
    try:
        context = retrieve(request.question, top_k=settings["top_k"])
        messages = build_prompt(
            system_prompt=settings["system_prompt"],
            context=context,
            history=request.history,
            question=request.question,
        )
        answer = ask_llm(
            messages,
            temperature=settings["temperature"],
            max_tokens=settings["max_new_tokens"],
        )
        return {"answer": answer, "context_used": context}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/settings")
def get_settings():
    return settings

@app.post("/settings")
def update_settings(request: SettingsRequest):
    settings.update(request.dict())
    return {"message": "Settings updated"}

@app.get("/health")
def health():
    return {"status": "ok"}
