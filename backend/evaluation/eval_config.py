import os

import config

BASE_DIR = os.path.dirname(__file__)
DEFAULT_GOLDEN_DATASET_PATH = os.path.join(BASE_DIR, "golden_dataset.json")
DEFAULT_REPORT_JSON_PATH = os.path.join(BASE_DIR, "report.json")
DEFAULT_REPORT_MD_PATH = os.path.join(BASE_DIR, "report.md")
DEFAULT_REPORT_YAML_PATH = os.path.join(BASE_DIR, "report_card.yaml")

EVAL_THRESHOLDS = {
    "context_precision": 0.85,
    "context_recall": 0.80,
    "context_relevancy": 0.75,
    "context_entity_recall": 0.70,
    "faithfulness": 0.85,
    "answer_relevancy": 0.80,
    "answer_correctness": 0.70,
    "answer_similarity": 0.75,
    "hallucination_rate": 0.15,
    "groundedness_score": 0.85,
    "retrieval_f1": 0.80,
    "overall_rag_score": 0.75,
}

LOWER_IS_BETTER = {"hallucination_rate"}

OVERALL_WEIGHTS = {
    "context_recall": 0.30,
    "context_precision": 0.20,
    "faithfulness": 0.25,
    "answer_relevancy": 0.15,
    "answer_correctness": 0.10,
}

EVALUATOR_LLM = os.getenv("EVALUATOR_LLM", "gpt-4o-mini")
EVALUATOR_EMBEDDING = os.getenv("EVALUATOR_EMBEDDING", "text-embedding-3-small")
EVALUATOR_OPENAI_BASE_URL = os.getenv("EVALUATOR_OPENAI_BASE_URL", config.DEFAULT_EMBEDDING_OPENAI_BASE_URL)
EVALUATOR_OPENAI_API_KEY = (
    os.getenv("EVALUATOR_OPENAI_API_KEY")
    or os.getenv("EMBEDDING_OPENAI_API_KEY")
    or os.getenv("OPENAI_API_KEY")
)

DEFAULT_EVALUATION_CONFIG = {
    "system_prompt": config.DEFAULT_SYSTEM_PROMPT,
    "temperature": 0.2,
    "max_new_tokens": config.MAX_NEW_TOKENS,
    "top_k": config.TOP_K,
    "evaluator_llm": EVALUATOR_LLM,
    "evaluator_embedding": EVALUATOR_EMBEDDING,
    "evaluator_openai_base_url": EVALUATOR_OPENAI_BASE_URL,
}
