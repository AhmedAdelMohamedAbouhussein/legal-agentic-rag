"""
metrics.py — Evaluation engine: Faithfulness, Cost, Latency.
"""

from __future__ import annotations
import json
import time
from dataclasses import dataclass, field
from typing import List

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

# ── Pricing (USD per 1M tokens, as of mid-2025) ───────────────────────────────
MODEL_PRICING = {
    "gpt-4o":           {"input": 5.00,  "output": 15.00},
    "gpt-4o-mini":      {"input": 0.15,  "output": 0.60},
    "gpt-3.5-turbo":    {"input": 0.50,  "output": 1.50},
    "gemini-1.5-pro":   {"input": 3.50,  "output": 10.50},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    "deepseek-chat":    {"input": 0.27,  "output": 1.10},
}

FAITHFULNESS_PROMPT = """You are a strict fact-checking judge for legal AI systems.

Task: Decide what fraction of the ANSWER's factual claims are directly supported by the CONTEXT.

ANSWER:
{answer}

CONTEXT (retrieved documents):
{context}

Instructions:
1. List each factual claim in the answer
2. Mark each as SUPPORTED or UNSUPPORTED based only on the context
3. Compute fraction = supported / total

Respond with valid JSON only:
{{
  "total_claims": <int>,
  "supported_claims": <int>,
  "faithfulness_score": <float 0.0-1.0>,
  "unsupported": ["<claim1>", ...]
}}
"""


@dataclass
class EvalResult:
    faithfulness: float = 0.0
    latency_seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    model: str = "gpt-4o-mini"
    unsupported_claims: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "faithfulness": round(self.faithfulness, 3),
            "latency_s": round(self.latency_seconds, 2),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "model": self.model,
        }


def evaluate_faithfulness(
    answer: str,
    docs: List[Document],
    llm: ChatOpenAI,
) -> tuple[float, List[str]]:
    """Return (faithfulness_score, unsupported_claims_list)."""
    context = "\n\n".join(d.page_content for d in docs)[:4000]
    prompt  = FAITHFULNESS_PROMPT.format(answer=answer, context=context)

    try:
        response = llm.invoke(prompt)
        text     = response.content
        start    = text.find("{")
        end      = text.rfind("}") + 1
        data     = json.loads(text[start:end])
        score    = float(data.get("faithfulness_score", 1.0))
        bad      = data.get("unsupported", [])
        return min(max(score, 0.0), 1.0), bad
    except Exception:
        return 1.0, []


def calculate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """Compute USD cost from token counts."""
    pricing = MODEL_PRICING.get(model, {"input": 5.00, "output": 15.00})
    cost = (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000
    return cost


class LatencyTimer:
    """Context manager for measuring elapsed seconds."""
    def __init__(self):
        self.elapsed = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed = time.perf_counter() - self._start
