import io
import json
import os
from datetime import datetime, timezone
from typing import Any

import pdfplumber
from chonkie import TokenChunker

import config
from llm import ask_llm, build_prompt


def generate_golden_dataset(
    pdf_path: str,
    output_path: str,
    questions_per_chunk: int = 1,
    source_document: str | None = None,
) -> dict:
    text = _extract_pdf_text(pdf_path)
    chunks = _chunk_text(text)
    samples = _generate_qa_from_chunks(chunks, questions_per_chunk=questions_per_chunk)

    dataset = {
        "metadata": {
            "source_document": source_document or os.path.basename(pdf_path),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "version": "1.0",
            "total_questions": len(samples),
            "chunk_size": config.CHUNK_SIZE,
            "chunk_overlap": config.CHUNK_OVERLAP,
        },
        "samples": samples,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(dataset, file, indent=2, ensure_ascii=False)

    return dataset


def _extract_pdf_text(pdf_path: str) -> str:
    with open(pdf_path, "rb") as file:
        file_bytes = file.read()

    text_parts = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n".join(text_parts)


def _chunk_text(text: str) -> list[str]:
    chunker = TokenChunker(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    return [chunk.text for chunk in chunker.chunk(text) if chunk.text.strip()]


def _generate_qa_from_chunks(chunks: list[str], questions_per_chunk: int = 1) -> list[dict]:
    samples = []
    for index, chunk in enumerate(chunks):
        qa_items = _generate_chunk_questions(chunk, questions_per_chunk=questions_per_chunk)
        for qa_index, qa_item in enumerate(qa_items):
            question = str(qa_item.get("question", "")).strip()
            ground_truth = str(qa_item.get("ground_truth", "")).strip()
            if not question or not ground_truth:
                continue
            classification = _classify_question(question)
            samples.append(
                {
                    "id": f"q_{len(samples) + 1:03d}",
                    "question": question,
                    "ground_truth": ground_truth,
                    "expected_contexts": [chunk],
                    "metadata": {
                        "category": classification["category"],
                        "difficulty": classification["difficulty"],
                        "source_chunk_index": index,
                        "source_page": [],
                        "topic": qa_item.get("topic", "general"),
                        "qa_index_in_chunk": qa_index,
                    },
                }
            )
    return samples


def _generate_chunk_questions(chunk: str, questions_per_chunk: int = 1) -> list[dict[str, Any]]:
    system_prompt = (
        "You generate high-quality golden evaluation data for a RAG chatbot. "
        "Return only valid JSON. Each answer must be fully supported by the provided context."
    )
    user_prompt = (
        f"Create {questions_per_chunk} evaluation question-answer pairs from this context. "
        "Use this JSON shape: "
        '[{"question":"...","ground_truth":"...","topic":"..."}]\n\n'
        f"Context:\n{chunk}"
    )
    messages = build_prompt(system_prompt=system_prompt, context="", history=[], question=user_prompt)
    raw_response = ask_llm(messages, temperature=0.1, max_tokens=900)
    return _parse_json_list(raw_response)


def _parse_json_list(raw_response: str) -> list[dict[str, Any]]:
    cleaned = raw_response.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return []

    if isinstance(parsed, dict):
        for key in ("samples", "questions", "items"):
            if isinstance(parsed.get(key), list):
                parsed = parsed[key]
                break

    if not isinstance(parsed, list):
        return []

    return [item for item in parsed if isinstance(item, dict)]


def _classify_question(question: str) -> dict:
    lowered = question.lower()
    if any(marker in lowered for marker in ("why", "how", "compare", "relationship", "difference")):
        return {"category": "reasoning", "difficulty": "medium"}
    if any(marker in lowered for marker in ("summarize", "overview", "main points")):
        return {"category": "summary", "difficulty": "medium"}
    return {"category": "factual", "difficulty": "easy"}
