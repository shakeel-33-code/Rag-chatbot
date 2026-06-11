import os
import tempfile

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import config
from observability import (
    record_exception,
    set_attribute,
    set_input,
    set_output,
    setup_observability,
    start_span,
    timed_span,
)
from ingest import ingest_pdf
from retriever import retrieve
from llm import build_prompt, ask_llm
from evaluation.eval_config import (
    DEFAULT_GOLDEN_DATASET_PATH,
    DEFAULT_REPORT_JSON_PATH,
    DEFAULT_REPORT_MD_PATH,
)
from evaluation.golden_dataset_generator import generate_golden_dataset
from evaluation.metrics import run_evaluation
from evaluation.report import load_latest_report

app = FastAPI()
setup_observability(app)

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

class EvaluationRunRequest(BaseModel):
    golden_dataset_path: str = DEFAULT_GOLDEN_DATASET_PATH
    output_path: str = DEFAULT_REPORT_JSON_PATH
    markdown_output_path: str = DEFAULT_REPORT_MD_PATH
    top_k: int | None = None

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    file_bytes = await file.read()
    with start_span(
        "rag.ingest",
        {
            "upload.filename": file.filename,
            "upload.file_size_bytes": len(file_bytes),
        },
        span_kind="CHAIN",
    ) as span:
        try:
            chunks_ingested = ingest_pdf(file_bytes, file.filename)
            set_attribute(span, "ingest.chunks_ingested", chunks_ingested)
        except Exception as e:
            record_exception(span, e)
            raise

    return {"message": f"Ingested {chunks_ingested} chunks from {file.filename}"}

@app.post("/chat")
def chat(request: ChatRequest):
    with start_span(
        "User Query",
        {
            "rag.question": request.question,
            "rag.history_length": len(request.history),
            "rag.top_k": settings["top_k"],
            "rag.temperature": settings["temperature"],
            "rag.max_new_tokens": settings["max_new_tokens"],
        },
        span_kind="CHAIN",
    ) as span:
        try:
            set_input(
                span,
                {
                    "question": request.question,
                    "history": request.history,
                    "settings": settings,
                },
                mime_type="application/json",
            )

            rewritten_question = preprocess_query(request.question, request.history)
            context = retrieve(rewritten_question, top_k=settings["top_k"])
            set_attribute(span, "rag.context_chars", len(context))

            messages = build_prompt(
                system_prompt=settings["system_prompt"],
                context=context,
                history=request.history,
                question=rewritten_question,
            )
            answer = ask_llm(
                messages,
                temperature=settings["temperature"],
                max_tokens=settings["max_new_tokens"],
            )
            final_answer = postprocess_response(answer)
            set_attribute(span, "rag.answer_chars", len(final_answer))
            set_output(
                span,
                {
                    "answer": final_answer,
                    "rewritten_question": rewritten_question,
                    "context_chars": len(context),
                },
                mime_type="application/json",
            )
            return {"answer": final_answer, "context_used": context}
        except Exception as e:
            record_exception(span, e)
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/settings")
def get_settings():
    return settings

@app.post("/settings")
def update_settings(request: SettingsRequest):
    settings.update(request.dict())
    return {"message": "Settings updated"}

@app.post("/evaluation/generate-golden-dataset")
async def generate_evaluation_dataset(file: UploadFile = File(...), questions_per_chunk: int = 1):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    file_bytes = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_pdf_path = temp_file.name

    try:
        dataset = generate_golden_dataset(
            pdf_path=temp_pdf_path,
            output_path=DEFAULT_GOLDEN_DATASET_PATH,
            questions_per_chunk=questions_per_chunk,
            source_document=file.filename,
        )
        return {
            "message": "Golden dataset generated",
            "path": DEFAULT_GOLDEN_DATASET_PATH,
            "total_questions": dataset["metadata"]["total_questions"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)

@app.post("/evaluation/run")
def run_evaluation_endpoint(request: EvaluationRunRequest = EvaluationRunRequest()):
    config_overrides = {
        "system_prompt": settings["system_prompt"],
        "temperature": settings["temperature"],
        "max_new_tokens": settings["max_new_tokens"],
        "top_k": request.top_k if request.top_k is not None else settings["top_k"],
    }
    try:
        report = run_evaluation(
            golden_dataset_path=request.golden_dataset_path,
            config=config_overrides,
            output_json_path=request.output_path,
            output_markdown_path=request.markdown_output_path,
        )
        return report.to_dict()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/evaluation/report")
def get_evaluation_report():
    try:
        return load_latest_report(DEFAULT_REPORT_JSON_PATH)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok"}


def preprocess_query(question: str, history: List[Dict[str, str]]) -> str:
    with timed_span(
        "preprocess/query_rewrite",
        "preprocess.query_rewrite.duration_ms",
        {
            "preprocess.history_messages": len(history),
            "preprocess.original_query_chars": len(question),
        },
        span_kind="CHAIN",
    ) as span:
        rewritten_question = question.strip()
        set_attribute(span, "preprocess.rewrite_applied", rewritten_question != question)
        set_attribute(span, "preprocess.rewritten_query_chars", len(rewritten_question))
        set_input(
            span,
            {"question": question, "history_messages": len(history)},
            mime_type="application/json",
        )
        set_output(span, {"rewritten_question": rewritten_question}, mime_type="application/json")
        return rewritten_question


def postprocess_response(answer: str) -> str:
    with timed_span(
        "response_postprocess",
        "response_postprocess.duration_ms",
        {"response.raw_chars": len(answer)},
        span_kind="CHAIN",
    ) as span:
        final_answer = answer.strip()
        set_attribute(span, "response.trimmed_chars", len(answer) - len(final_answer))
        set_attribute(span, "response.final_chars", len(final_answer))
        set_input(span, answer, mime_type="text/plain")
        set_output(span, final_answer, mime_type="text/plain")
        return final_answer
