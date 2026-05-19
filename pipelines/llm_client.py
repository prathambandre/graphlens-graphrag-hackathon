"""
Unified LLM Client for GraphLens.
Supports: Groq (free), OpenAI, Google Gemini, and Mock mode.
Tracks token usage, latency, and cost per call.
"""
import time
import random
import hashlib
from typing import Dict, Any, Optional

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config


class LLMClient:
    """Unified LLM client with provider abstraction and metric tracking."""

    def __init__(self, provider: Optional[str] = None):
        self.provider = provider or (Config.LLM_PROVIDER if not Config.is_mock() else "mock")
        self._client = None
        self._init_client()

    def _init_client(self):
        """Initialize the appropriate LLM client."""
        if self.provider == "mock":
            return

        if self.provider == "groq":
            try:
                from groq import Groq
                self._client = Groq(api_key=Config.GROQ_API_KEY)
            except ImportError:
                print("[WARN] groq package not installed, falling back to mock")
                self.provider = "mock"

        elif self.provider == "openai":
            try:
                import openai
                self._client = openai.OpenAI(api_key=Config.OPENAI_API_KEY)
            except ImportError:
                print("[WARN] openai package not installed, falling back to mock")
                self.provider = "mock"

        elif self.provider == "gemini":
            try:
                import google.generativeai as genai
                genai.configure(api_key=Config.GEMINI_API_KEY)
                self._client = genai
            except ImportError:
                print("[WARN] google-generativeai package not installed, falling back to mock")
                self.provider = "mock"

    def generate(self, prompt: str, system_prompt: str = "", temperature: float = 0.3) -> Dict[str, Any]:
        """
        Generate a response from the LLM.
        Returns: {text, input_tokens, output_tokens, latency_ms, cost_usd, provider}
        """
        start = time.perf_counter()

        if self.provider == "mock":
            result = self._mock_generate(prompt, system_prompt)
        elif self.provider == "groq":
            result = self._groq_generate(prompt, system_prompt, temperature)
        elif self.provider == "openai":
            result = self._openai_generate(prompt, system_prompt, temperature)
        elif self.provider == "gemini":
            result = self._gemini_generate(prompt, system_prompt, temperature)
        else:
            result = self._mock_generate(prompt, system_prompt)

        elapsed = (time.perf_counter() - start) * 1000  # ms
        result["latency_ms"] = round(elapsed, 2)
        result["cost_usd"] = self._calc_cost(result["input_tokens"], result["output_tokens"])
        result["provider"] = self.provider
        return result

    def _mock_generate(self, prompt: str, system_prompt: str) -> Dict[str, Any]:
        """Generate a realistic mock response based on query content."""
        # Simulate processing time
        time.sleep(random.uniform(0.3, 0.8))

        # Generate deterministic but varied responses based on prompt hash
        seed = int(hashlib.md5(prompt.encode()).hexdigest()[:8], 16)
        rng = random.Random(seed)

        # Estimate input tokens (rough approximation: 1 token ~ 4 chars)
        input_tokens = len(prompt) // 4 + len(system_prompt) // 4

        # Determine if this has context (RAG) or not
        has_context = "Context:" in prompt or "Retrieved" in prompt
        has_graph = "Graph" in prompt or "entities" in prompt.lower() or "relationships" in prompt.lower()

        if has_graph:
            # GraphRAG-style response: precise, structured, fewer tokens
            output_tokens = rng.randint(120, 200)
            response = self._generate_graph_response(prompt, rng)
        elif has_context:
            # Basic RAG response: broader, more tokens
            output_tokens = rng.randint(180, 350)
            response = self._generate_rag_response(prompt, rng)
        else:
            # LLM-Only response: verbose, potentially hallucinated
            output_tokens = rng.randint(250, 500)
            response = self._generate_llm_only_response(prompt, rng)

        return {
            "text": response,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }

    def _generate_llm_only_response(self, prompt: str, rng: random.Random) -> str:
        """Generate a plausible LLM-only response (no retrieval context)."""
        query = prompt.split("Question:")[-1].strip() if "Question:" in prompt else prompt[:200]
        templates = [
            f"Based on my training data, I can provide some general information about this topic. {query.split('?')[0] if '?' in query else query} involves several key aspects. "
            f"While I don't have access to specific source documents, the general understanding is that this relates to fundamental concepts in the domain. "
            f"It's important to note that there may be more specific details available in specialized databases or recent publications that I may not have access to. "
            f"The topic typically encompasses multiple interconnected factors that contribute to the overall understanding. "
            f"However, without access to specific reference materials, I should note that some details may not be fully accurate or up-to-date.",

            f"From my general knowledge, this query touches on several important areas. {query.split('?')[0] if '?' in query else query} is a topic that has been extensively studied. "
            f"The key points to understand are that this involves complex interactions between multiple factors. "
            f"Various experts have proposed different frameworks for understanding this, and the consensus view suggests multiple contributing elements. "
            f"I should caveat that my response is based on general training data and may not reflect the most current information available. "
            f"For the most accurate and detailed information, consulting specialized sources would be recommended.",
        ]
        return rng.choice(templates)

    def _generate_rag_response(self, prompt: str, rng: random.Random) -> str:
        """Generate a plausible Basic RAG response."""
        query = prompt.split("Question:")[-1].strip() if "Question:" in prompt else prompt[:200]
        templates = [
            f"Based on the retrieved documents, I can provide the following information. "
            f"According to the source materials, {query.split('?')[0] if '?' in query else query} is documented across several relevant passages. "
            f"The retrieved text chunks indicate that this topic is well-covered in the knowledge base. "
            f"Key findings from the retrieved context include relevant factual information that addresses the core of the query. "
            f"The retrieved passages provide substantial coverage of the topic with moderate specificity. "
            f"Sources: [chunk_12, chunk_45, chunk_78, chunk_123, chunk_201]",

            f"The retrieved context provides relevant information for this query. "
            f"From the vector-matched passages, we can determine that {query.split('?')[0] if '?' in query else query} relates to concepts documented in the knowledge base. "
            f"The semantic search identified 5 relevant passages that address different aspects of the query. "
            f"While the retrieved chunks provide good topical coverage, some passages may include tangentially related information due to the nature of vector similarity search. "
            f"Sources: [chunk_34, chunk_67, chunk_89, chunk_156, chunk_234]",
        ]
        return rng.choice(templates)

    def _generate_graph_response(self, prompt: str, rng: random.Random) -> str:
        """Generate a plausible GraphRAG response."""
        query = prompt.split("Question:")[-1].strip() if "Question:" in prompt else prompt[:200]
        templates = [
            f"Through graph-based retrieval with multi-hop traversal, I identified the precise entities and relationships relevant to this query. "
            f"{query.split('?')[0] if '?' in query else query} connects to key entities through well-defined relationship paths in the knowledge graph. "
            f"The 2-hop traversal revealed 3 directly connected entities with 5 relationship edges providing structured context. "
            f"This graph-grounded approach ensures the response is factually anchored to verified entity relationships rather than broad text similarity. "
            f"Evidence path: Entity_A -[RELATES_TO]-> Entity_B -[PART_OF]-> Entity_C",

            f"The knowledge graph traversal identified a precise subgraph relevant to this query. "
            f"Starting from the extracted entities, a 2-hop BFS traversal through the TigerGraph knowledge graph revealed structured relationships that directly address the question. "
            f"Unlike vector-only retrieval, the graph context captures the exact relationship chain: the query entities connect through 4 intermediate nodes with typed edges. "
            f"This structured retrieval reduces irrelevant context by 65% compared to basic RAG while maintaining 95% recall on the relevant information. "
            f"Graph path: [Query Entity] -> [Connected Entity (2 hops)] -> [Answer Entity]",
        ]
        return rng.choice(templates)

    def _groq_generate(self, prompt: str, system_prompt: str, temperature: float) -> Dict[str, Any]:
        """Generate using Groq API."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self._client.chat.completions.create(
            model=Config.GROQ_MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=1024,
        )
        return {
            "text": response.choices[0].message.content,
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        }

    def _openai_generate(self, prompt: str, system_prompt: str, temperature: float) -> Dict[str, Any]:
        """Generate using OpenAI API."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self._client.chat.completions.create(
            model=Config.OPENAI_MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=1024,
        )
        return {
            "text": response.choices[0].message.content,
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        }

    def _gemini_generate(self, prompt: str, system_prompt: str, temperature: float) -> Dict[str, Any]:
        """Generate using Google Gemini API."""
        model = self._client.GenerativeModel(Config.GEMINI_MODEL)
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        response = model.generate_content(
            full_prompt,
            generation_config={"temperature": temperature, "max_output_tokens": 1024}
        )
        # Rough token estimation for Gemini
        input_tokens = len(full_prompt) // 4
        output_tokens = len(response.text) // 4
        return {
            "text": response.text,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }

    def _calc_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost in USD."""
        in_cost = input_tokens * Config.get_cost_per_token("input")
        out_cost = output_tokens * Config.get_cost_per_token("output")
        return round(in_cost + out_cost, 6)
