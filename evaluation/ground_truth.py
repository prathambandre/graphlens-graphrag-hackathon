"""
Ground Truth Benchmark Questions for GraphLens.
30 questions across 3 difficulty levels designed to test multi-hop reasoning.
"""

GROUND_TRUTH = [
    # === SINGLE-HOP (Easy) - 10 questions ===
    {
        "id": "q01",
        "query": "What is a knowledge graph and how does it store information?",
        "expected_answer": "A knowledge graph is a structured representation of information that stores data as entities (nodes) and relationships (edges) between them. It organizes knowledge in a graph format where nodes represent real-world objects, concepts, or events, and edges represent the semantic relationships between these entities. This allows for rich, interconnected data representation.",
        "difficulty": "easy",
        "requires_multi_hop": False,
        "category": "definitions",
    },
    {
        "id": "q02",
        "query": "What is Retrieval-Augmented Generation (RAG)?",
        "expected_answer": "Retrieval-Augmented Generation (RAG) is an AI framework that enhances Large Language Model responses by first retrieving relevant information from an external knowledge base before generating an answer. It combines information retrieval with text generation to produce more accurate, grounded, and up-to-date responses.",
        "difficulty": "easy",
        "requires_multi_hop": False,
        "category": "definitions",
    },
    {
        "id": "q03",
        "query": "What is TigerGraph and what type of database is it?",
        "expected_answer": "TigerGraph is a native parallel graph database platform designed for enterprise-scale graph analytics. It is a graph database that uses a graph-based data model to store, query, and analyze highly interconnected data. It supports real-time deep-link analytics, pattern matching, and machine learning on graph data.",
        "difficulty": "easy",
        "requires_multi_hop": False,
        "category": "technology",
    },
    {
        "id": "q04",
        "query": "What is FAISS and what is it used for?",
        "expected_answer": "FAISS (Facebook AI Similarity Search) is an open-source library developed by Meta for efficient similarity search and clustering of dense vectors. It is commonly used in RAG systems to create vector indexes that enable fast nearest-neighbor search over embedding vectors, allowing retrieval of semantically similar text chunks.",
        "difficulty": "easy",
        "requires_multi_hop": False,
        "category": "technology",
    },
    {
        "id": "q05",
        "query": "What is BERTScore and how does it evaluate text quality?",
        "expected_answer": "BERTScore is an automatic evaluation metric for text generation that computes similarity scores between candidate and reference texts using BERT contextual embeddings. It measures precision, recall, and F1 score at the token level using cosine similarity of BERT embeddings, providing a more semantically meaningful evaluation than traditional n-gram based metrics.",
        "difficulty": "easy",
        "requires_multi_hop": False,
        "category": "evaluation",
    },
    {
        "id": "q06",
        "query": "What is entity extraction in NLP?",
        "expected_answer": "Entity extraction (Named Entity Recognition - NER) is an NLP task that identifies and classifies named entities in text into predefined categories such as person names, organizations, locations, dates, and quantities. It is a fundamental step in knowledge graph construction, extracting structured information from unstructured text.",
        "difficulty": "easy",
        "requires_multi_hop": False,
        "category": "definitions",
    },
    {
        "id": "q07",
        "query": "What are vector embeddings in the context of machine learning?",
        "expected_answer": "Vector embeddings are numerical representations of data (text, images, etc.) in a continuous vector space where semantically similar items are mapped to nearby points. In NLP, text embeddings capture semantic meaning, enabling similarity comparisons between documents or sentences using distance metrics like cosine similarity.",
        "difficulty": "easy",
        "requires_multi_hop": False,
        "category": "definitions",
    },
    {
        "id": "q08",
        "query": "What is the role of a context window in LLM inference?",
        "expected_answer": "The context window is the maximum number of tokens an LLM can process in a single inference call, including both the input prompt and the generated output. It determines how much information can be provided to the model at once. Managing context window size is crucial for efficiency, as larger contexts increase token usage, latency, and cost.",
        "difficulty": "easy",
        "requires_multi_hop": False,
        "category": "technology",
    },
    {
        "id": "q09",
        "query": "What is semantic search and how does it differ from keyword search?",
        "expected_answer": "Semantic search uses natural language understanding and vector embeddings to find results based on meaning rather than exact keyword matches. Unlike keyword search which looks for literal term overlap, semantic search captures conceptual similarity, understanding synonyms, context, and intent to return more relevant results.",
        "difficulty": "easy",
        "requires_multi_hop": False,
        "category": "definitions",
    },
    {
        "id": "q10",
        "query": "What is the LLM-as-a-Judge evaluation method?",
        "expected_answer": "LLM-as-a-Judge is an evaluation method where a powerful LLM is used to assess the quality of responses generated by AI systems. The judge LLM scores responses on criteria like factual accuracy, completeness, relevance, and hallucination, providing automated qualitative evaluation that correlates with human judgment.",
        "difficulty": "easy",
        "requires_multi_hop": False,
        "category": "evaluation",
    },

    # === MULTI-HOP (Medium) - 12 questions ===
    {
        "id": "q11",
        "query": "How does GraphRAG reduce token usage compared to basic vector RAG, and why does this lead to cost savings?",
        "expected_answer": "GraphRAG reduces token usage by retrieving only the most precisely relevant context through graph traversal rather than broad vector similarity matches. By following entity relationships in a knowledge graph, it identifies smaller, more targeted context chunks. Since LLM API costs are directly proportional to token count, fewer input tokens per query translate to lower per-query costs, typically achieving 60-80% token reduction.",
        "difficulty": "medium",
        "requires_multi_hop": True,
        "category": "comparison",
    },
    {
        "id": "q12",
        "query": "What is multi-hop reasoning in knowledge graphs and why is it important for question answering?",
        "expected_answer": "Multi-hop reasoning involves traversing multiple edges in a knowledge graph to connect information across several intermediate entities. For question answering, it enables the system to answer complex queries that require combining facts from different but connected pieces of information. For example, answering 'What university did the CEO of Company X attend?' requires following the path: Company X -> CEO -> Person -> Education -> University.",
        "difficulty": "medium",
        "requires_multi_hop": True,
        "category": "reasoning",
    },
    {
        "id": "q13",
        "query": "Compare the retrieval strategies of FAISS-based RAG and TigerGraph-based GraphRAG in terms of precision and recall.",
        "expected_answer": "FAISS-based RAG uses vector similarity to retrieve the top-k most semantically similar text chunks, which provides high recall but may include tangentially related content (lower precision). TigerGraph-based GraphRAG uses structured entity relationships and graph traversal, providing higher precision by following explicit relationship paths, though it may miss relevant information not captured in the graph structure (potentially lower recall). The combination yields better precision-per-token efficiency.",
        "difficulty": "medium",
        "requires_multi_hop": True,
        "category": "comparison",
    },
    {
        "id": "q14",
        "query": "Explain the ingestion pipeline for building a knowledge graph from unstructured documents.",
        "expected_answer": "The ingestion pipeline involves: 1) Document parsing and text extraction, 2) Text chunking into manageable segments, 3) Named Entity Recognition (NER) to extract entities, 4) Relationship extraction to identify connections between entities, 5) Entity resolution and deduplication, 6) Upserting entities as nodes and relationships as edges into the graph database, and 7) Creating entity-to-chunk links to maintain provenance. This transforms unstructured text into a structured knowledge graph.",
        "difficulty": "medium",
        "requires_multi_hop": True,
        "category": "architecture",
    },
    {
        "id": "q15",
        "query": "How does the LLM-as-a-Judge method handle evaluation of multi-hop reasoning quality specifically?",
        "expected_answer": "LLM-as-a-Judge evaluates multi-hop reasoning by assessing whether the response correctly chains multiple pieces of information together. The judge checks if the answer follows a logical path connecting related facts, whether intermediate reasoning steps are valid, and if the final conclusion is supported by the chain of evidence. It specifically penalizes responses that skip logical steps or make unsupported leaps.",
        "difficulty": "medium",
        "requires_multi_hop": True,
        "category": "evaluation",
    },
    {
        "id": "q16",
        "query": "What are the trade-offs between using local embeddings (sentence-transformers) versus API-based embeddings (OpenAI) for RAG?",
        "expected_answer": "Local embeddings (sentence-transformers) offer zero API cost, no rate limits, lower latency, and data privacy since text never leaves the local machine. However, they may have lower quality than state-of-the-art API models and require local GPU/CPU resources. API-based embeddings (OpenAI) provide higher quality representations but incur per-token costs, depend on network availability, have rate limits, and require sending data externally.",
        "difficulty": "medium",
        "requires_multi_hop": True,
        "category": "technology",
    },
    {
        "id": "q17",
        "query": "How does TigerGraph's native parallel processing benefit GraphRAG query latency compared to traditional graph databases?",
        "expected_answer": "TigerGraph's native parallel graph architecture processes graph traversals using massively parallel computation, distributing BFS/DFS operations across multiple processing units simultaneously. This reduces multi-hop traversal time from O(n) sequential processing to near-constant time for bounded-depth queries. For GraphRAG, this means faster entity neighborhood retrieval, enabling real-time multi-hop context gathering that keeps total query latency competitive.",
        "difficulty": "medium",
        "requires_multi_hop": True,
        "category": "technology",
    },
    {
        "id": "q18",
        "query": "Why might a basic RAG system hallucinate more than a GraphRAG system on complex queries?",
        "expected_answer": "Basic RAG retrieves chunks based on surface-level semantic similarity, which may include tangentially related but factually irrelevant content. When the LLM processes this noisy context, it may conflate information from unrelated chunks, leading to hallucination. GraphRAG retrieves context through verified entity relationships, ensuring the provided information is structurally connected and factually related, reducing the chance of the LLM combining unrelated facts.",
        "difficulty": "medium",
        "requires_multi_hop": True,
        "category": "comparison",
    },
    {
        "id": "q19",
        "query": "Describe how BERTScore and LLM-as-a-Judge complement each other in evaluating RAG system outputs.",
        "expected_answer": "BERTScore provides an automated, reproducible quantitative measure of semantic similarity between generated and reference texts, but it only captures surface-level semantic overlap. LLM-as-a-Judge provides qualitative assessment of factual accuracy, reasoning quality, and hallucination detection that BERTScore cannot measure. Together, they offer both a reliable numerical baseline (BERTScore) and a nuanced quality assessment (Judge), giving a comprehensive evaluation picture.",
        "difficulty": "medium",
        "requires_multi_hop": True,
        "category": "evaluation",
    },
    {
        "id": "q20",
        "query": "How does the chunking strategy affect both vector RAG retrieval quality and knowledge graph construction?",
        "expected_answer": "Chunking strategy critically affects both pipelines. For vector RAG, smaller chunks increase retrieval precision but may lose context, while larger chunks maintain context but reduce precision and increase token usage. For knowledge graph construction, chunk size affects entity co-occurrence detection - too small may miss relationships spanning multiple sentences, too large may create spurious entity connections. Optimal chunking (typically 200-500 tokens with overlap) balances both needs.",
        "difficulty": "medium",
        "requires_multi_hop": True,
        "category": "architecture",
    },
    {
        "id": "q21",
        "query": "What role does entity resolution play in building an accurate knowledge graph for GraphRAG?",
        "expected_answer": "Entity resolution identifies and merges references to the same real-world entity that appear with different surface forms (e.g., 'TigerGraph', 'TG', 'Tiger Graph Inc.'). Without it, the knowledge graph would contain duplicate nodes, fragmenting the relationship network and weakening multi-hop traversal. Accurate entity resolution consolidates all relationships under canonical entity nodes, enabling complete traversal paths and accurate context retrieval.",
        "difficulty": "medium",
        "requires_multi_hop": True,
        "category": "architecture",
    },
    {
        "id": "q22",
        "query": "Compare the cost-per-query economics of LLM-Only, Basic RAG, and GraphRAG pipelines at scale.",
        "expected_answer": "LLM-Only has the lowest per-query token cost but produces lower quality answers requiring human verification. Basic RAG increases input tokens (retrieved chunks) but improves answer quality, with costs scaling linearly with chunk count and size. GraphRAG optimizes by retrieving fewer, more precise tokens through graph traversal, achieving similar or better quality than Basic RAG at 40-60% lower token cost. At scale (millions of queries), GraphRAG's token efficiency provides significant cumulative cost savings.",
        "difficulty": "medium",
        "requires_multi_hop": True,
        "category": "comparison",
    },

    # === COMPLEX REASONING (Hard) - 8 questions ===
    {
        "id": "q23",
        "query": "Design an optimal evaluation framework that would definitively prove GraphRAG superiority over basic RAG for enterprise use cases. What metrics, test cases, and methodology would you use?",
        "expected_answer": "An optimal framework would include: 1) Stratified test set with single-hop, multi-hop, and temporal reasoning questions, 2) Metrics: BERTScore F1 for semantic accuracy, LLM-as-a-Judge for factual correctness and hallucination (5-point scale), token efficiency ratio, latency percentiles (p50/p95/p99), cost-per-correct-answer, 3) A/B testing with statistical significance (p<0.05), 4) Human evaluation subset for calibrating automated metrics, 5) Stress testing under concurrent load, 6) Domain-specific evaluation across multiple verticals.",
        "difficulty": "hard",
        "requires_multi_hop": True,
        "category": "evaluation",
    },
    {
        "id": "q24",
        "query": "Explain how a hybrid retrieval strategy combining vector search and graph traversal could outperform either approach alone, and describe the implementation architecture.",
        "expected_answer": "A hybrid approach first uses vector search for broad semantic recall, then filters and re-ranks results using graph proximity scores. Implementation: 1) Embed query and retrieve top-k candidates via FAISS, 2) Extract entities from query using NER, 3) For each candidate chunk, compute graph distance to query entities via TigerGraph traversal, 4) Re-rank candidates using a weighted combination of vector similarity and graph proximity, 5) Select top-n results with highest combined score. This captures both semantic relevance and structural connectivity.",
        "difficulty": "hard",
        "requires_multi_hop": True,
        "category": "architecture",
    },
    {
        "id": "q25",
        "query": "What are the failure modes of GraphRAG systems and how can they be mitigated in production?",
        "expected_answer": "Key failure modes: 1) Incomplete graph: missing entities/relationships lead to failed traversals - mitigate with continuous ingestion and quality checks, 2) Entity extraction errors: wrong entities lead to irrelevant context - mitigate with ensemble NER and human-in-the-loop validation, 3) Graph sparsity: isolated subgraphs block multi-hop paths - mitigate with entity resolution and bridge entity detection, 4) Stale data: outdated graph doesn't reflect current knowledge - mitigate with incremental updates and version tracking, 5) Query-entity mismatch: query entities not in graph - mitigate with fallback to vector search.",
        "difficulty": "hard",
        "requires_multi_hop": True,
        "category": "reasoning",
    },
    {
        "id": "q26",
        "query": "How would you scale a GraphRAG system from handling 1M tokens to 1B tokens while maintaining query latency under 2 seconds?",
        "expected_answer": "Scaling strategies: 1) Distributed graph partitioning in TigerGraph across multiple nodes for parallel traversal, 2) Tiered storage: hot entities in memory, warm in SSD, cold in object storage, 3) Pre-computed entity embeddings with approximate nearest neighbor indexes for fast seed entity identification, 4) Query-time graph pruning: limit traversal depth and fan-out based on edge weights, 5) Caching frequently traversed subgraphs, 6) Async pipeline: parallelize entity extraction, graph traversal, and LLM generation, 7) Index materialization for common multi-hop patterns.",
        "difficulty": "hard",
        "requires_multi_hop": True,
        "category": "architecture",
    },
    {
        "id": "q27",
        "query": "Analyze the relationship between graph topology metrics (density, diameter, clustering coefficient) and GraphRAG retrieval quality.",
        "expected_answer": "Graph density affects context richness: higher density provides more traversal paths but may introduce noise. Graph diameter impacts multi-hop reach: smaller diameter means more entities are reachable within k hops, improving recall. Clustering coefficient affects context coherence: higher clustering means retrieved entities are more interconnected, producing more coherent context. Optimal GraphRAG performance requires balanced topology: moderate density (0.01-0.1), small diameter (4-8), and moderate clustering (0.3-0.6). Monitoring these metrics helps predict and tune retrieval quality.",
        "difficulty": "hard",
        "requires_multi_hop": True,
        "category": "reasoning",
    },
    {
        "id": "q28",
        "query": "Compare the end-to-end architectures of Microsoft's GraphRAG, TigerGraph's GraphRAG, and a custom FAISS+Neo4j solution for enterprise deployment.",
        "expected_answer": "Microsoft GraphRAG uses hierarchical community summarization with global/local search modes, best for broad summarization but expensive to build. TigerGraph GraphRAG leverages native parallel graph processing with hybrid vector+graph retrieval, offering superior query performance at scale due to its MPP architecture. A custom FAISS+Neo4j solution provides flexibility but requires manual orchestration of vector and graph queries, lacks native parallel traversal, and has higher operational complexity. TigerGraph excels for real-time multi-hop queries at scale; Microsoft's approach is better for global summarization tasks.",
        "difficulty": "hard",
        "requires_multi_hop": True,
        "category": "comparison",
    },
    {
        "id": "q29",
        "query": "How can graph-based context retrieval reduce LLM hallucination rates, and what empirical evidence supports this claim?",
        "expected_answer": "Graph-based retrieval reduces hallucination by: 1) Providing structurally verified context where entity relationships are explicit rather than inferred, 2) Limiting context to provenance-tracked information linked through known paths, 3) Enabling the LLM to cite specific entity-relationship chains. Empirical evidence from GraphRAG benchmarks shows 40-60% reduction in hallucination rates compared to vector-only RAG, measured via LLM-as-a-Judge hallucination scores and human evaluation. The structured nature of graph context constrains the LLM's generation space to factually grounded responses.",
        "difficulty": "hard",
        "requires_multi_hop": True,
        "category": "reasoning",
    },
    {
        "id": "q30",
        "query": "Design a real-time monitoring dashboard for a production GraphRAG system. What KPIs would you track and what alerting thresholds would you set?",
        "expected_answer": "Key KPIs: 1) Query latency (p50 < 1s, p99 < 3s, alert at p99 > 5s), 2) Token efficiency ratio (graph tokens / vector tokens, target < 0.5, alert > 0.8), 3) Graph traversal depth utilization (target 2-3 hops, alert if consistently hitting max depth), 4) Entity extraction confidence (target > 0.8, alert < 0.6), 5) Cache hit rate (target > 60%, alert < 30%), 6) BERTScore rolling average (target > 0.75, alert < 0.6), 7) Hallucination rate from periodic judge evaluation (target < 10%, alert > 25%), 8) Graph freshness (time since last ingestion, alert > 24h), 9) Cost per query (target < $0.002, alert > $0.01).",
        "difficulty": "hard",
        "requires_multi_hop": True,
        "category": "architecture",
    },
]


def get_questions(difficulty=None, category=None):
    """Filter ground truth questions by difficulty and/or category."""
    questions = GROUND_TRUTH
    if difficulty:
        questions = [q for q in questions if q["difficulty"] == difficulty]
    if category:
        questions = [q for q in questions if q["category"] == category]
    return questions


def get_question_by_id(qid):
    """Get a specific question by ID."""
    for q in GROUND_TRUTH:
        if q["id"] == qid:
            return q
    return None
