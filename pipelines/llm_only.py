"""
Pipeline 1: LLM-Only
Direct query to LLM with no retrieval context.
Serves as the baseline for comparison.
"""
import time
from typing import Dict, Any

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipelines.llm_client import LLMClient


SYSTEM_PROMPT = """You are a knowledgeable AI assistant. Answer the user's question to the best of your ability based solely on your training knowledge. If you are unsure about something, clearly state your uncertainty. Be concise and factual."""


class LLMOnlyPipeline:
    """Pipeline 1: Direct LLM inference with no retrieval."""

    def __init__(self):
        self.llm = LLMClient()
        self.name = "LLM-Only"

    def query(self, question: str) -> Dict[str, Any]:
        """Run a query through the LLM-only pipeline."""
        start = time.perf_counter()

        prompt = f"Question: {question}\n\nProvide a clear, accurate answer."

        result = self.llm.generate(prompt, system_prompt=SYSTEM_PROMPT)

        total_latency = (time.perf_counter() - start) * 1000

        return {
            "pipeline": self.name,
            "answer": result["text"],
            "input_tokens": result["input_tokens"],
            "output_tokens": result["output_tokens"],
            "total_tokens": result["input_tokens"] + result["output_tokens"],
            "latency_ms": round(total_latency, 2),
            "retrieval_latency_ms": 0,
            "generation_latency_ms": result["latency_ms"],
            "cost_usd": result["cost_usd"],
            "provider": result["provider"],
            "context_chunks": 0,
            "graph_hops": 0,
            "entities_found": 0,
            "sources": [],
        }
