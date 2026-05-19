"""
GraphLens Configuration
Loads settings from .env or uses defaults (mock mode).
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Central configuration for GraphLens."""

    # Mode
    MODE = os.getenv("MODE", "mock")  # "mock" or "live"

    # LLM Provider
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    # TigerGraph
    TG_HOST = os.getenv("TG_HOST", "")
    TG_USERNAME = os.getenv("TG_USERNAME", "tigergraph")
    TG_PASSWORD = os.getenv("TG_PASSWORD", "")
    TG_GRAPH_NAME = os.getenv("TG_GRAPH_NAME", "GraphLens")
    TG_USE_MOCK = os.getenv("TG_USE_MOCK", "true").lower() == "true"

    # Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data", "corpus")
    FAISS_INDEX_PATH = os.path.join(BASE_DIR, "data", "faiss_index")
    METRICS_DB_PATH = os.path.join(BASE_DIR, "data", "metrics.db")

    # Pipeline settings
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    CHUNK_SIZE = 500  # tokens per chunk
    CHUNK_OVERLAP = 50
    TOP_K = 5  # retrieval count
    GRAPH_HOPS = 2  # multi-hop traversal depth

    # Cost per 1M tokens (for estimation)
    COST_TABLE = {
        "groq": {"input": 0.05, "output": 0.10},
        "openai": {"input": 0.15, "output": 0.60},
        "gemini": {"input": 0.075, "output": 0.30},
        "ollama": {"input": 0.00, "output": 0.00},
        "mock": {"input": 0.05, "output": 0.10},
    }

    @classmethod
    def is_mock(cls):
        return cls.MODE == "mock"

    @classmethod
    def get_cost_per_token(cls, direction="input"):
        provider = cls.LLM_PROVIDER if not cls.is_mock() else "mock"
        rate = cls.COST_TABLE.get(provider, cls.COST_TABLE["mock"])
        return rate[direction] / 1_000_000
