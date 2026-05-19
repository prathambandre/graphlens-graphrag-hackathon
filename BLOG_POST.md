The RAG Secret Nobody Tells You: Why Vector Search is Blowing Your LLM Budget (And How Graphs Fix It)

A hands-on case study from the TigerGraph GraphRAG Inference Hackathon.

By Pratham Bandre & Vinit Prajapati

---

We have all been there. You build a Retrieval-Augmented Generation (RAG) system, ask it a complex question, and wait. And wait. When the answer finally arrives, it is either a generic summary that missed the point, or it cost you five times more tokens than it should have.

When we started building GraphLens for the TigerGraph GraphRAG Inference Hackathon, we wanted to address a simple question: 

Why does standard vector search feel so inefficient for complex queries?

After building three different inference pipelines side-by-side (LLM-Only, Basic Vector RAG, and TigerGraph-powered GraphRAG) and testing them on a 2-million-token dataset, we found the answer. And the numbers surprised us.

---

The Problem: Vector Search Drowns Your LLM in Noise

Vector similarity search is excellent for finding matching terms or single facts. If you ask, "What is Retrieval-Augmented Generation?", a vector database like FAISS easily finds the exact chunk defining it.

But real-world queries are rarely that simple. They require connecting the dots. 

Consider a question like: 
"How does TigerGraph's parallel processing benefit GraphRAG query latency?"

To answer this, an LLM needs to understand:
1. What is TigerGraph's architecture?
2. What is GraphRAG?
3. How do they connect?
4. What is the impact on latency?

When you run this through a standard vector search pipeline, it retrieves the top-5 most semantically similar paragraphs. Because it lacks structural context, it returns broad, verbose text chunks that mention parallel processing, graph databases, or latency in completely different contexts.

This results in:
- Token Bloat: Your LLM prompt is suddenly stuffed with 2000+ tokens of noisy text.
- Higher Costs: Since API costs scale linearly with token counts, you are paying for the database's lack of precision.
- Hallucinations: When forced to synthesize answers from disconnected paragraphs, LLMs often mix up unrelated facts.

---

The Solution: Enter GraphRAG with TigerGraph

Instead of scanning the entire corpus for similar-sounding sentences, GraphLens extracts the key entities from the query and traverses a structured knowledge graph built on TigerGraph.

How GraphLens works under the hood:

1. Entity Extraction: The query is processed to find core concepts (like TigerGraph, parallel processing, latency).
2. Multi-Hop Traversal: We perform a 2-hop traversal in TigerGraph, starting from the query entities to find direct relationships.
3. Structured Context Assembly: We feed the LLM a clean, structured list of verified relationships (e.g., TigerGraph -[FEATURES]-> Parallel_Processing -[REDUCES]-> Query_Latency) along with only the text chunks directly linked to those relationships.
4. Focused Generation: The LLM synthesizes a precise answer based on verified relationship paths rather than guesses.

[Insert System Architecture Image here: https://raw.githubusercontent.com/prathambandre/graphlens-graphrag-hackathon/main/assets/architecture_diagram.png]

---

The Experiment: Head-to-Head Comparison

We built a Streamlit comparison dashboard to benchmark three configurations on the same hardware and dataset:

- Pipeline 1 (LLM-Only): Direct query to the LLM (no retrieval context).
- Pipeline 2 (Basic RAG): Traditional semantic search using FAISS and local embeddings.
- Pipeline 3 (GraphRAG): Multi-hop retrieval using TigerGraph.

[Insert Dashboard Screenshot here: https://raw.githubusercontent.com/prathambandre/graphlens-graphrag-hackathon/main/assets/dashboard_comparison_results.png]

---

The Results: The Data Speaks for Itself

We evaluated 30 complex multi-hop questions across three difficulty levels using two main metrics: BERTScore F1 (semantic accuracy against ground truth answers) and LLM-as-a-Judge (grading factual accuracy, completeness, and hallucination on a 1-5 scale).

Here is what we observed:

1. Token Efficiency (The Cost Saver)
GraphRAG used 67% fewer tokens per query compared to Basic Vector RAG (average of 600 tokens vs 1800 tokens). By retrieving only the exact entities and relationships needed, we cut out the fluff. At scale, this reduces API costs by more than half.

2. Answer Quality & Accuracy
- Factual Grounding: GraphRAG achieved an average LLM Judge score of 4.3/5, compared to 3.5/5 for Basic RAG and 2.5/5 for LLM-Only.
- Multi-Hop Accuracy: On questions requiring logical connections across multiple documents, GraphRAG reached 78% accuracy while Basic Vector RAG struggled at 35%.
- Zero Hallucination: Because the prompt context is constrained to verified graph edges, the LLM consistently cited its exact path sources rather than making up connections.

---

Why Graphs Beat Vectors for Complex Queries

The core takeaway is simple: Precision beats recall.

Vector search is a wide net. It gives you high recall (you probably retrieved the answer somewhere in those 2000 tokens), but low precision (most of it was noise). 

Graph traversal is a targeted laser. By tracing the exact relationships between entities, it delivers high precision. This means the LLM gets a cleaner prompt, responds faster, costs less, and does not hallucinate.

---

Try GraphLens Yourself

We have open-sourced the entire project, including the Streamlit dashboard, benchmarking suite, and fallback pipelines so you can run it locally without setup friction:

- GitHub Repository: https://github.com/prathambandre/graphlens-graphrag-hackathon
- Run it locally: Clone the repository, install requirements.txt, and run python run.py.

We are excited to expand this for Round 2 by integrating live TigerGraph Cloud instances and exploring hybrid vector-graph layouts. 

If you are building in the RAG space, let us know your thoughts in the comments below!

---

Built for the TigerGraph GraphRAG Inference Hackathon 2026
#GraphRAGInferenceHackathon @TigerGraph

If you found this useful, give the repo a star and follow us for updates on Round 2!
