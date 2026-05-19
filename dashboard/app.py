"""
GraphLens - Comparison Dashboard
Premium Streamlit UI for side-by-side pipeline comparison.
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import time
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipelines.llm_only import LLMOnlyPipeline
from pipelines.basic_rag import BasicRAGPipeline
from pipelines.graph_rag import GraphRAGPipeline
from evaluation.evaluator import Evaluator
from evaluation.ground_truth import GROUND_TRUTH, get_questions

# --- Page Config ---
st.set_page_config(page_title="GraphLens | GraphRAG Inference", layout="wide", initial_sidebar_state="expanded", page_icon="🔬")

# --- CSS ---
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }
.main { background: linear-gradient(135deg, #0f0c29 0%, #1a1a3e 40%, #24243e 100%); }
[data-testid="stSidebar"] { background: rgba(15,12,41,0.95); border-right: 1px solid rgba(139,92,246,0.2); }
h1,h2,h3 { background: linear-gradient(90deg, #a78bfa, #818cf8, #6366f1); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }

.pipeline-card {
    background: rgba(30,27,75,0.6); backdrop-filter: blur(16px);
    border: 1px solid rgba(139,92,246,0.25); border-radius: 16px;
    padding: 1.5rem; margin: 0.5rem 0; transition: all 0.3s ease;
}
.pipeline-card:hover { border-color: rgba(139,92,246,0.6); transform: translateY(-2px); box-shadow: 0 8px 32px rgba(139,92,246,0.15); }
.pipeline-card h3 { font-size: 1.1rem !important; margin-bottom: 0.5rem; }

.metric-strip {
    background: rgba(30,27,75,0.4); backdrop-filter: blur(12px);
    border: 1px solid rgba(99,102,241,0.2); border-radius: 12px;
    padding: 1rem; text-align: center; transition: all 0.3s ease;
}
.metric-strip:hover { border-color: rgba(99,102,241,0.5); }
.metric-value { font-size: 1.8rem; font-weight: 700; color: #a78bfa; }
.metric-label { font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; }

.hero-badge {
    display: inline-block; padding: 4px 14px; border-radius: 20px; font-size: 0.7rem;
    font-weight: 600; letter-spacing: 1px; text-transform: uppercase; margin: 2px;
}
.badge-llm { background: rgba(239,68,68,0.15); color: #f87171; border: 1px solid rgba(239,68,68,0.3); }
.badge-rag { background: rgba(251,191,36,0.15); color: #fbbf24; border: 1px solid rgba(251,191,36,0.3); }
.badge-graph { background: rgba(52,211,153,0.15); color: #34d399; border: 1px solid rgba(52,211,153,0.3); }

.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important; border: none !important; border-radius: 12px !important;
    padding: 0.6rem 2rem !important; font-weight: 600 !important;
    transition: all 0.3s ease !important; font-family: 'Outfit', sans-serif !important;
}
.stButton > button:hover { transform: translateY(-2px) !important; box-shadow: 0 8px 25px rgba(99,102,241,0.4) !important; }

.answer-box {
    background: rgba(15,12,41,0.5); border-radius: 12px; padding: 1rem;
    border-left: 3px solid; margin-top: 0.5rem; font-size: 0.9rem; line-height: 1.6;
}
.answer-llm { border-left-color: #f87171; }
.answer-rag { border-left-color: #fbbf24; }
.answer-graph { border-left-color: #34d399; }
</style>""", unsafe_allow_html=True)


@st.cache_resource
def load_pipelines():
    return LLMOnlyPipeline(), BasicRAGPipeline(), GraphRAGPipeline()

@st.cache_resource
def load_evaluator():
    return Evaluator()


def render_header():
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("# 🔬 GraphLens")
        st.markdown("##### Efficient Graph-Powered RAG for Precision Retrieval")
        st.caption("1 Query → 3 Pipelines → Side-by-Side Comparison | TigerGraph GraphRAG Inference Hackathon")
    with col2:
        st.markdown("""
        <div style="text-align:right; padding-top:1rem;">
            <span class="hero-badge badge-llm">LLM-Only</span>
            <span class="hero-badge badge-rag">Basic RAG</span>
            <span class="hero-badge badge-graph">GraphRAG</span>
        </div>""", unsafe_allow_html=True)


def render_metrics_strip(results):
    """Render the KPI metric strip."""
    r_llm, r_rag, r_graph = results
    cols = st.columns(5)

    metrics = [
        ("Total Tokens", f"{r_graph['total_tokens']}", f"{((r_rag['total_tokens']-r_graph['total_tokens'])/max(r_rag['total_tokens'],1)*100):.0f}% fewer vs RAG", "inverse"),
        ("Latency", f"{r_graph['latency_ms']:.0f}ms", f"GraphRAG", "off"),
        ("Cost/Query", f"${r_graph['cost_usd']:.4f}", f"${r_rag['cost_usd']:.4f} RAG", "inverse"),
        ("Context Chunks", f"{r_graph['context_chunks']}", f"vs {r_rag['context_chunks']} in RAG", "inverse"),
        ("Graph Hops", f"{r_graph['graph_hops']}", f"{r_graph.get('entities_found',0)} entities", "off"),
    ]
    for i, (label, value, delta, delta_color) in enumerate(metrics):
        with cols[i]:
            st.markdown(f"""<div class="metric-strip">
                <div class="metric-value">{value}</div>
                <div class="metric-label">{label}</div>
                <div style="font-size:0.7rem;color:#64748b;margin-top:4px;">{delta}</div>
            </div>""", unsafe_allow_html=True)


def render_answer_cards(results):
    """Render side-by-side answer cards."""
    configs = [
        ("LLM-Only", "badge-llm", "answer-llm", "#f87171"),
        ("Basic RAG", "badge-rag", "answer-rag", "#fbbf24"),
        ("GraphRAG", "badge-graph", "answer-graph", "#34d399"),
    ]
    cols = st.columns(3)
    for i, (result, (name, badge, ans_cls, color)) in enumerate(zip(results, configs)):
        with cols[i]:
            st.markdown(f"""<div class="pipeline-card">
                <h3><span class="hero-badge {badge}">{name}</span></h3>
                <div class="answer-box {ans_cls}">{result['answer'][:500]}{'...' if len(result['answer'])>500 else ''}</div>
                <div style="margin-top:0.8rem;display:flex;justify-content:space-between;font-size:0.75rem;color:#94a3b8;">
                    <span>🎯 {result['total_tokens']} tokens</span>
                    <span>⚡ {result['latency_ms']:.0f}ms</span>
                    <span>💰 ${result['cost_usd']:.5f}</span>
                </div>
                <div style="font-size:0.7rem;color:#64748b;margin-top:4px;">
                    Sources: {', '.join(result['sources'][:3]) if result['sources'] else 'None (parametric only)'}
                </div>
            </div>""", unsafe_allow_html=True)


def render_comparison_charts(results):
    """Render Plotly comparison charts."""
    r_llm, r_rag, r_graph = results
    pipelines = ["LLM-Only", "Basic RAG", "GraphRAG"]
    colors = ["#f87171", "#fbbf24", "#34d399"]

    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure(data=[go.Bar(
            x=pipelines, y=[r_llm['total_tokens'], r_rag['total_tokens'], r_graph['total_tokens']],
            marker_color=colors, text=[r_llm['total_tokens'], r_rag['total_tokens'], r_graph['total_tokens']],
            textposition='auto',
        )])
        fig.update_layout(
            title="Token Usage Comparison", template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Outfit"), height=350, margin=dict(t=50,b=30,l=30,r=30),
            yaxis_title="Tokens",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure(data=[go.Bar(
            x=pipelines, y=[r_llm['latency_ms'], r_rag['latency_ms'], r_graph['latency_ms']],
            marker_color=colors, text=[f"{r_llm['latency_ms']:.0f}", f"{r_rag['latency_ms']:.0f}", f"{r_graph['latency_ms']:.0f}"],
            textposition='auto',
        )])
        fig.update_layout(
            title="Latency Comparison (ms)", template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Outfit"), height=350, margin=dict(t=50,b=30,l=30,r=30),
            yaxis_title="Milliseconds",
        )
        st.plotly_chart(fig, use_container_width=True)


def render_eval_charts(eval_results):
    """Render evaluation score charts."""
    if not eval_results:
        return

    col1, col2 = st.columns(2)

    with col1:
        # Radar chart for judge scores
        categories = ['Accuracy', 'Completeness', 'Anti-Hallucination', 'Reasoning']
        fig = go.Figure()
        colors_map = {"LLM-Only": "#f87171", "Basic RAG": "#fbbf24", "GraphRAG": "#34d399"}
        for pname, evals in eval_results.items():
            j = evals["judge_scores"]
            vals = [j["factual_accuracy"], j["completeness"], j["hallucination"], j["reasoning_quality"]]
            fig.add_trace(go.Scatterpolar(
                r=vals + [vals[0]], theta=categories + [categories[0]],
                fill='toself', name=pname, line=dict(color=colors_map.get(pname, "#fff")),
                fillcolor=colors_map.get(pname, "#fff").replace(")", ",0.1)").replace("rgb", "rgba") if "rgb" in colors_map.get(pname, "") else None,
                opacity=0.8,
            ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 5], gridcolor="rgba(255,255,255,0.1)"),
                       bgcolor="rgba(0,0,0,0)", angularaxis=dict(gridcolor="rgba(255,255,255,0.1)")),
            title="LLM-as-a-Judge Scores (1-5)", template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Outfit"), height=400,
            margin=dict(t=60,b=30,l=60,r=60), showlegend=True,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # BERTScore comparison
        pipelines = list(eval_results.keys())
        f1_scores = [eval_results[p]["bert_score"]["f1"] for p in pipelines]
        colors = ["#f87171", "#fbbf24", "#34d399"]
        fig = go.Figure(data=[go.Bar(
            x=pipelines, y=f1_scores, marker_color=colors[:len(pipelines)],
            text=[f"{s:.3f}" for s in f1_scores], textposition='auto',
        )])
        fig.update_layout(
            title="BERTScore F1 Comparison", template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Outfit"), height=400, margin=dict(t=50,b=30,l=30,r=30),
            yaxis_title="F1 Score", yaxis=dict(range=[0, 1]),
        )
        st.plotly_chart(fig, use_container_width=True)


def render_benchmark_tab():
    """Full benchmark evaluation across all ground truth questions."""
    st.markdown("### 📊 Full Benchmark Report")
    st.caption("Run evaluation across all 30 ground truth questions with BERTScore + LLM-as-a-Judge")

    p1, p2, p3 = load_pipelines()
    evaluator = load_evaluator()

    difficulty = st.selectbox("Filter by difficulty", ["all", "easy", "medium", "hard"])
    questions = get_questions(difficulty=None if difficulty == "all" else difficulty)

    if st.button(f"🚀 Run Benchmark ({len(questions)} questions)", key="run_bench"):
        progress = st.progress(0)
        status = st.empty()
        all_results = []

        for idx, q in enumerate(questions):
            status.text(f"Evaluating: {q['query'][:60]}...")
            progress.progress((idx + 1) / len(questions))

            r1 = p1.query(q["query"])
            r2 = p2.query(q["query"])
            r3 = p3.query(q["query"])

            e1 = evaluator.evaluate_pipeline_result(q["query"], r1["answer"], q["expected_answer"])
            e2 = evaluator.evaluate_pipeline_result(q["query"], r2["answer"], q["expected_answer"])
            e3 = evaluator.evaluate_pipeline_result(q["query"], r3["answer"], q["expected_answer"])

            all_results.append({
                "Question": q["query"][:80] + "...",
                "Difficulty": q["difficulty"],
                "Multi-hop": "Yes" if q["requires_multi_hop"] else "No",
                "LLM Tokens": r1["total_tokens"],
                "RAG Tokens": r2["total_tokens"],
                "Graph Tokens": r3["total_tokens"],
                "LLM BERTScore": e1["bert_score"]["f1"],
                "RAG BERTScore": e2["bert_score"]["f1"],
                "Graph BERTScore": e3["bert_score"]["f1"],
                "LLM Judge": e1["judge_scores"]["overall"],
                "RAG Judge": e2["judge_scores"]["overall"],
                "Graph Judge": e3["judge_scores"]["overall"],
                "Token Savings": f"{((r2['total_tokens']-r3['total_tokens'])/max(r2['total_tokens'],1)*100):.1f}%",
            })

        status.text("Benchmark complete!")
        df = pd.DataFrame(all_results)
        st.dataframe(df, use_container_width=True, height=400)

        # Summary stats
        st.markdown("---")
        st.markdown("### 📈 Aggregate Results")
        summary_cols = st.columns(3)
        for i, (pname, prefix, color) in enumerate([("LLM-Only","LLM","#f87171"),("Basic RAG","RAG","#fbbf24"),("GraphRAG","Graph","#34d399")]):
            with summary_cols[i]:
                avg_bert = df[f"{prefix} BERTScore"].mean()
                avg_judge = df[f"{prefix} Judge"].mean()
                avg_tokens = df[f"{prefix} Tokens"].mean()
                st.markdown(f"""<div class="pipeline-card">
                    <h3 style="color:{color}!important;-webkit-text-fill-color:{color}!important;">{pname}</h3>
                    <p>Avg BERTScore F1: <b>{avg_bert:.3f}</b></p>
                    <p>Avg Judge Score: <b>{avg_judge:.1f}/5</b></p>
                    <p>Avg Tokens: <b>{avg_tokens:.0f}</b></p>
                </div>""", unsafe_allow_html=True)

        # Download
        csv = df.to_csv(index=False)
        st.download_button("📥 Download Results CSV", csv, "graphlens_benchmark.csv", "text/csv")


def render_architecture_tab():
    """Show architecture diagram and system description."""
    st.markdown("### 🏗️ System Architecture")
    st.markdown("""
    <div class="pipeline-card">
        <h3>GraphLens Architecture Overview</h3>
        <pre style="color:#a78bfa;font-size:0.8rem;line-height:1.8;overflow-x:auto;">
┌─────────────────────────────────────────────────────────────┐
│                    STREAMLIT DASHBOARD                       │
│  ┌──────────┐   ┌──────────────┐   ┌────────────────────┐  │
│  │ LLM-Only │   │  Basic RAG   │   │ GraphRAG (TigerG.) │  │
│  │ Pipeline  │   │  Pipeline    │   │ Pipeline           │  │
│  └─────┬─────┘   └──────┬───────┘   └────────┬───────────┘  │
│        │                │                     │              │
│  ┌─────┴─────────────────┴─────────────────────┴──────────┐  │
│  │            BENCHMARK METRICS PANEL                      │  │
│  │  Tokens │ Latency │ Cost │ LLM Judge │ BERTScore       │  │
│  └─────────────────────────────────────────────────────────┘  │
└────────────────────────────┬────────────────────────────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
    ┌────┴─────┐      ┌─────┴──────┐     ┌──────┴───────┐
    │ Direct   │      │   FAISS    │     │  TigerGraph  │
    │ LLM Call │      │  Vector    │     │  Knowledge   │
    │          │      │  Index     │     │  Graph (KG)  │
    │ No       │      │            │     │              │
    │ retrieval│      │ Semantic   │     │ Entity +     │
    │          │      │ similarity │     │ Multi-hop    │
    └──────────┘      └────────────┘     └──────────────┘
        </pre>
    </div>""", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""<div class="pipeline-card">
            <h3><span class="hero-badge badge-llm">Pipeline 1</span> LLM-Only</h3>
            <p style="color:#94a3b8;font-size:0.85rem;">Direct query to LLM with zero retrieval. Relies entirely on parametric knowledge. Baseline for comparison.</p>
            <p style="color:#64748b;font-size:0.75rem;">Pros: Fastest, lowest token usage<br>Cons: Hallucination risk, no sources</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="pipeline-card">
            <h3><span class="hero-badge badge-rag">Pipeline 2</span> Basic RAG</h3>
            <p style="color:#94a3b8;font-size:0.85rem;">FAISS vector search retrieves top-k similar chunks. Context-augmented LLM generation. Standard approach.</p>
            <p style="color:#64748b;font-size:0.75rem;">Pros: Grounded answers, citable<br>Cons: Noisy context, high tokens</p>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="pipeline-card">
            <h3><span class="hero-badge badge-graph">Pipeline 3</span> GraphRAG</h3>
            <p style="color:#94a3b8;font-size:0.85rem;">TigerGraph knowledge graph with entity extraction + multi-hop BFS traversal. Structured, precise context.</p>
            <p style="color:#64748b;font-size:0.75rem;">Pros: Precise, fewer tokens, multi-hop<br>Cons: Requires graph construction</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("### 🛠️ Tech Stack")
    tech = pd.DataFrame({
        "Component": ["Graph DB", "Vector Store", "Embeddings", "NER", "LLM", "Evaluation", "Dashboard"],
        "Technology": ["TigerGraph (NetworkX fallback)", "FAISS (faiss-cpu)", "sentence-transformers (all-MiniLM-L6-v2)", "spaCy (en_core_web_sm)", "Groq / OpenAI / Gemini (Mock mode)", "BERTScore + LLM-as-a-Judge", "Streamlit + Plotly"],
        "Role": ["Knowledge graph storage & traversal", "Semantic similarity search", "Local text embeddings (free)", "Entity extraction from queries", "Answer generation", "Quality measurement", "Interactive comparison UI"],
    })
    st.dataframe(tech, use_container_width=True, hide_index=True)


def main():
    render_header()
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        st.markdown(f"**Mode:** {'🟡 Mock (Demo)' if True else '🟢 Live'}")
        st.markdown("**Team:** Pratham Bandre & Vinit Prajapati")
        st.markdown("---")
        st.markdown("### 📋 Sample Questions")
        sample_qs = [
            "What is a knowledge graph?",
            "How does GraphRAG reduce token usage compared to basic RAG?",
            "Compare FAISS-based RAG and TigerGraph GraphRAG precision",
            "Explain multi-hop reasoning in knowledge graphs",
            "Design an optimal evaluation framework for GraphRAG",
        ]
        selected_q = None
        for q in sample_qs:
            if st.button(q[:50] + "...", key=f"sq_{hash(q)}", use_container_width=True):
                selected_q = q
        st.markdown("---")
        st.markdown("### 🏆 Hackathon")
        st.markdown("**TigerGraph GraphRAG**")
        st.markdown("**Inference Hackathon**")
        st.markdown("#GraphRAGInferenceHackathon")

    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Query Comparison", "📊 Benchmark Report", "🏗️ Architecture"])

    with tab1:
        query = st.text_input("Enter your question:", value=selected_q or "", placeholder="e.g., How does GraphRAG improve LLM inference?", key="main_query")

        if st.button("⚡ Compare All Pipelines", key="compare_btn", use_container_width=True):
            if query.strip():
                p1, p2, p3 = load_pipelines()
                evaluator = load_evaluator()

                with st.spinner("Running all 3 pipelines..."):
                    r1 = p1.query(query)
                    r2 = p2.query(query)
                    r3 = p3.query(query)

                results = [r1, r2, r3]
                st.markdown("### 📊 Performance Metrics")
                render_metrics_strip(results)
                st.markdown("### 💬 Answers")
                render_answer_cards(results)
                st.markdown("### 📈 Comparison Charts")
                render_comparison_charts(results)

                # Evaluation
                st.markdown("### 🎯 Quality Evaluation")
                ref = ""
                for gq in GROUND_TRUTH:
                    if gq["query"].lower() in query.lower() or query.lower() in gq["query"].lower():
                        ref = gq["expected_answer"]
                        break

                eval_results = {}
                for name, result in zip(["LLM-Only", "Basic RAG", "GraphRAG"], results):
                    ev = evaluator.evaluate_pipeline_result(query, result["answer"], ref)
                    eval_results[name] = ev

                render_eval_charts(eval_results)

                # Graph details
                if r3.get("graph_stats"):
                    with st.expander("🔗 Graph Traversal Details"):
                        gs = r3["graph_stats"]
                        st.write(f"**Seed Entities:** {', '.join(gs['seed_entities'])}")
                        st.write(f"**Nodes Traversed:** {gs['nodes_traversed']}")
                        st.write(f"**Edges Traversed:** {gs['edges_traversed']}")
                        if gs.get("relationships"):
                            st.markdown("**Relationship Paths:**")
                            for rel in gs["relationships"]:
                                st.code(f"{rel['source']} -[{rel['type']}]-> {rel['target']} (hop {rel['hop']})")
            else:
                st.warning("Please enter a question to compare.")

    with tab2:
        render_benchmark_tab()

    with tab3:
        render_architecture_tab()

    # Footer
    st.markdown("---")
    st.markdown("""<div style="text-align:center;color:#64748b;font-size:0.75rem;padding:1rem 0;">
        <b>GraphLens</b> | Built by Pratham Bandre & Vinit Prajapati |
        TigerGraph GraphRAG Inference Hackathon 2026 |
        <a href="https://github.com" style="color:#818cf8;">GitHub</a>
    </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
