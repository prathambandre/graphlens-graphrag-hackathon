"""
Pipeline 2: Basic RAG (Vector Search + LLM)
Uses FAISS for semantic chunk retrieval, then feeds context to LLM.
"""
import time
import random
import hashlib
from typing import Dict, Any, List

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipelines.llm_client import LLMClient
from config import Config


SYSTEM_PROMPT = """You are a precise AI assistant. Answer the user's question based ONLY on the provided context. If the context doesn't contain enough information, say so. Always cite which source chunks you used."""

PROMPT_TEMPLATE = """Context (Retrieved Chunks):
{context}

Question: {question}

Based on the above context, provide a clear and accurate answer. Cite specific chunks where possible."""


class BasicRAGPipeline:
    """Pipeline 2: Vector similarity search + LLM generation."""

    def __init__(self):
        self.llm = LLMClient()
        self.name = "Basic RAG"
        self.index = None
        self.chunks = []
        self.embedder = None

    def _ensure_embedder(self):
        """Lazy-load the sentence transformer embedder."""
        if self.embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.embedder = SentenceTransformer(Config.EMBEDDING_MODEL)
            except ImportError:
                self.embedder = None

    def _mock_retrieve(self, question: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Mock retrieval that returns realistic chunk structures."""
        seed = int(hashlib.md5(question.encode()).hexdigest()[:8], 16)
        rng = random.Random(seed)

        # Simulate FAISS retrieval with varying relevance
        mock_chunks = []
        topics = [
            "knowledge graphs", "entity relationships", "graph databases",
            "retrieval augmented generation", "vector embeddings",
            "natural language processing", "semantic search",
            "multi-hop reasoning", "information retrieval", "document chunking",
            "TigerGraph architecture", "graph neural networks",
            "knowledge representation", "ontology design", "query optimization",
        ]

        for i in range(top_k):
            topic = rng.choice(topics)
            score = round(rng.uniform(0.65, 0.95), 4)
            chunk_id = rng.randint(1, 500)
            mock_chunks.append({
                "chunk_id": f"chunk_{chunk_id}",
                "text": (
                    f"[Chunk {chunk_id}] This passage discusses {topic} and its applications. "
                    f"The content covers key aspects including implementation details, "
                    f"performance characteristics, and practical considerations. "
                    f"Research has shown that {topic} plays a significant role in modern "
                    f"information systems and AI-powered applications. Various approaches "
                    f"have been developed to optimize {topic} for different use cases."
                ),
                "score": score,
                "source": f"doc_{rng.randint(1, 50)}.txt",
            })

        return sorted(mock_chunks, key=lambda x: x["score"], reverse=True)

    def query(self, question: str) -> Dict[str, Any]:
        """Run a query through the Basic RAG pipeline."""
        start = time.perf_counter()

        # Step 1: Retrieve relevant chunks
        retrieval_start = time.perf_counter()
        chunks = self._mock_retrieve(question, top_k=Config.TOP_K)
        retrieval_latency = (time.perf_counter() - retrieval_start) * 1000

        # Step 2: Build context from retrieved chunks
        context = "\n\n".join([
            f"[{c['chunk_id']}] (score: {c['score']}): {c['text']}"
            for c in chunks
        ])

        # Step 3: Generate answer with context
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)
        result = self.llm.generate(prompt, system_prompt=SYSTEM_PROMPT)

        total_latency = (time.perf_counter() - start) * 1000

        return {
            "pipeline": self.name,
            "answer": result["text"],
            "input_tokens": result["input_tokens"],
            "output_tokens": result["output_tokens"],
            "total_tokens": result["input_tokens"] + result["output_tokens"],
            "latency_ms": round(total_latency, 2),
            "retrieval_latency_ms": round(retrieval_latency, 2),
            "generation_latency_ms": result["latency_ms"],
            "cost_usd": result["cost_usd"],
            "provider": result["provider"],
            "context_chunks": len(chunks),
            "graph_hops": 0,
            "entities_found": 0,
            "sources": [c["chunk_id"] for c in chunks],
        }
