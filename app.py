"""
app.py — Legal AI Assistant | Agentic RAG Dashboard
"""
import os, uuid
import streamlit as st
from dotenv import load_dotenv

load_dotenv()


# ── LLM factory ───────────────────────────────────────────────────────────────
def _get_llm(model, oai_key, gem_key="", ds_key=""):
    if model.startswith("gemini") and gem_key:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model, google_api_key=gem_key, temperature=0.1)
    if model.startswith("deepseek") and ds_key:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, openai_api_key=ds_key,
                          openai_api_base="https://api.deepseek.com/v1", temperature=0.1)
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=model, openai_api_key=oai_key, temperature=0.1)


# ── Query runner ──────────────────────────────────────────────────────────────
def _run_query(question, model, oai_key, gem_key, ds_key):
    from evaluator.metrics import evaluate_faithfulness, calculate_cost, LatencyTimer
    from langchain_openai import ChatOpenAI

    agent  = st.session_state.agent
    thread = {"configurable": {"thread_id": st.session_state.thread_id}}

    st.session_state.chat.append({"role": "user", "content": question})

    with LatencyTimer() as timer:
        state = agent.invoke(
            {
                "question":      question,
                "chat_history":  [m for m in st.session_state.chat[:-1]],
                "steps":         [],
                "loop_count":    0,
                "is_sufficient": False,
                "retrieved_docs": [],
                "search_queries": [],
                "query_type":    "both",
                "answer":        "",
            },
            config=thread,
        )

    answer  = state.get("answer", "No answer generated.")
    steps   = state.get("steps", [])
    docs    = state.get("retrieved_docs", [])
    sources = list({d.metadata.get("filename", "doc") for d in docs})

    eval_llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=oai_key, temperature=0)
    faith, _ = evaluate_faithfulness(answer, docs, eval_llm)

    in_tok  = len(question.split()) * 4
    out_tok = len(answer.split()) * 4
    cost    = calculate_cost(model, in_tok, out_tok)
    st.session_state.total_cost  += cost
    st.session_state.query_count += 1

    st.session_state.chat.append({
        "role": "assistant", "content": answer,
        "sources": sources, "steps": steps,
    })
    st.session_state.eval_log.append({
        "query":        question[:60],
        "faithfulness": faith,
        "latency_s":    timer.elapsed,
        "cost_usd":     cost,
        "model":        model,
    })

st.set_page_config(
    page_title="Legal AI Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg, #0a0a1a 0%, #1a0a2e 40%, #0a1a2e 100%); color: #e2e8f0; }
section[data-testid="stSidebar"] { background: rgba(255,255,255,0.03); border-right: 1px solid rgba(255,255,255,0.07); }
.glass { background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); border-radius: 16px; padding: 20px; backdrop-filter: blur(10px); margin-bottom: 16px; }
.metric-card { background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); border-radius: 12px; padding: 14px 18px; text-align: center; }
.metric-card .val { font-size: 1.9rem; font-weight: 700; background: linear-gradient(135deg, #667eea, #f093fb); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.metric-card .lbl { font-size: 0.72rem; color: #94a3b8; margin-top: 2px; }
.user-msg { background: linear-gradient(135deg, #667eea, #764ba2); border-radius: 18px 18px 4px 18px; padding: 12px 18px; margin: 8px 0; max-width: 80%; margin-left: auto; color: white; font-size: 0.95rem; }
.ai-msg { background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.12); border-radius: 18px 18px 18px 4px; padding: 12px 18px; margin: 8px 0; max-width: 85%; color: #e2e8f0; font-size: 0.95rem; }
.step-item { font-size: 0.82rem; color: #94a3b8; padding: 4px 0; border-left: 2px solid rgba(102,126,234,0.4); padding-left: 10px; margin: 4px 0; }
.src-badge { display:inline-block; background:rgba(102,126,234,0.2); border:1px solid rgba(102,126,234,0.4); color:#a5b4fc; border-radius:999px; padding:2px 10px; font-size:0.72rem; margin:2px; }
.case-card { background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08); border-radius:12px; padding:16px; margin-bottom:12px; }
.tag { display:inline-block; border-radius:6px; padding:2px 8px; font-size:0.7rem; font-weight:600; margin-right:4px; }
.tag-cl { background:rgba(59,130,246,0.2); color:#93c5fd; border:1px solid rgba(59,130,246,0.3); }
.tag-ct { background:rgba(16,185,129,0.2); color:#6ee7b7; border:1px solid rgba(16,185,129,0.3); }
.tag-bt { background:rgba(245,158,11,0.2); color:#fcd34d; border:1px solid rgba(245,158,11,0.3); }
.stButton > button { background: linear-gradient(135deg,#667eea,#764ba2); color:white; border:none; border-radius:10px; font-weight:600; transition:all 0.2s; width:100%; }
.stButton > button:hover { transform:translateY(-1px); box-shadow:0 6px 20px rgba(102,126,234,0.4); }
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
def init():
    for k, v in {
        "indexed": False, "chat": [], "eval_log": [],
        "agent": None, "stores": None, "thread_id": str(uuid.uuid4()),
        "total_cost": 0.0, "query_count": 0,
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v

init()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<h1 style='font-size:1.4rem;font-weight:700;'>⚖️ Legal AI Assistant</h1>"
                "<p style='color:#94a3b8;font-size:0.8rem;margin-bottom:20px;'>Agentic RAG • LangGraph • Multi-LLM</p>",
                unsafe_allow_html=True)

    st.markdown("<div style='font-size:0.68rem;font-weight:600;letter-spacing:.1em;text-transform:uppercase;color:#667eea;margin-bottom:6px;'>🔑 API Keys</div>", unsafe_allow_html=True)
    openai_key = st.text_input("OpenAI API Key", value=os.getenv("OPENAI_API_KEY",""), type="password", placeholder="sk-...")
    gemini_key = st.text_input("Google Gemini Key (optional)", value=os.getenv("GOOGLE_API_KEY",""), type="password", placeholder="AIza...")
    deepseek_key = st.text_input("DeepSeek Key (optional)", value=os.getenv("DEEPSEEK_API_KEY",""), type="password", placeholder="sk-...")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.68rem;font-weight:600;letter-spacing:.1em;text-transform:uppercase;color:#667eea;margin-bottom:6px;'>🤖 Model</div>", unsafe_allow_html=True)
    model_options = {"GPT-4o": "gpt-4o", "GPT-4o-mini (fast)": "gpt-4o-mini", "GPT-3.5-turbo": "gpt-3.5-turbo"}
    if gemini_key:
        model_options["Gemini 1.5 Pro"] = "gemini-1.5-pro"
        model_options["Gemini 1.5 Flash"] = "gemini-1.5-flash"
    if deepseek_key:
        model_options["DeepSeek Chat"] = "deepseek-chat"
    selected_label = st.selectbox("Active Model", list(model_options.keys()))
    selected_model = model_options[selected_label]

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.68rem;font-weight:600;letter-spacing:.1em;text-transform:uppercase;color:#667eea;margin-bottom:6px;'>✂️ Chunking</div>", unsafe_allow_html=True)
    strategy = st.selectbox("Strategy", ["recursive","overlapping","fixed","token"],
                            format_func=lambda s: {"recursive":"🔀 Recursive","overlapping":"🔁 Overlapping","fixed":"📐 Fixed","token":"🪙 Token"}[s])
    chunk_size = st.slider("Chunk Size", 300, 3000, 1000, 100)
    chunk_overlap = st.slider("Overlap", 0, 500, 150, 50)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.session_state.indexed:
        st.markdown("<div style='background:rgba(16,185,129,0.1);border:1px solid rgba(16,185,129,0.3);border-radius:8px;padding:10px;color:#6ee7b7;font-size:0.82rem;'>✅ Documents indexed & ready</div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑️ Reset & Re-index"):
            for k in ["indexed","chat","eval_log","agent","stores","thread_id","total_cost","query_count"]:
                del st.session_state[k]
            init()
            st.rerun()
    else:
        if st.button("⚡ Index Legal Documents"):
            if not openai_key:
                st.error("OpenAI API key required.")
            else:
                with st.spinner("Indexing legal documents…"):
                    try:
                        from rag import load_documents_from_folder, chunk_documents, get_embeddings, build_vector_stores
                        from agent import build_agent
                        from langchain_openai import ChatOpenAI

                        docs_path = os.path.join(os.path.dirname(__file__), "legal_docs")
                        docs = load_documents_from_folder(docs_path)

                        if not docs:
                            st.error("No documents found in legal_docs/")
                        else:
                            chunks  = chunk_documents(docs, strategy=strategy, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                            emb     = get_embeddings(openai_key)
                            persist = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
                            cl_store, ct_store, gn_store = build_vector_stores(chunks, emb, persist)
                            st.session_state.stores = (cl_store, ct_store, gn_store)

                            llm = _get_llm(selected_model, openai_key, gemini_key, deepseek_key)
                            agent = build_agent(
                                cl_store.as_retriever(search_kwargs={"k": 5}),
                                ct_store.as_retriever(search_kwargs={"k": 5}),
                                gn_store.as_retriever(search_kwargs={"k": 5}),
                                llm,
                            )
                            st.session_state.agent   = agent
                            st.session_state.indexed = True
                            st.success(f"✅ Indexed {len(docs)} documents → {len(chunks)} chunks")
                            st.rerun()
                    except Exception as e:
                        st.error(f"Indexing error: {e}")


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style='font-size:2.4rem;font-weight:700;margin-bottom:4px;'>⚖️ Legal AI Assistant</h1>
<p style='color:#94a3b8;margin-bottom:28px;'>Agentic RAG powered by LangGraph • Multi-LLM • Real-time Evaluation</p>
""", unsafe_allow_html=True)

# Metric row
c1,c2,c3,c4 = st.columns(4)
metrics = [
    (st.session_state.query_count, "Queries"),
    (len(st.session_state.chat) // 2, "Exchanges"),
    (f"${st.session_state.total_cost:.4f}", "Est. Cost"),
    ("✅" if st.session_state.indexed else "⏳", "Status"),
]
for col, (val, lbl) in zip([c1,c2,c3,c4], metrics):
    with col:
        st.markdown(f"<div class='metric-card'><div class='val'>{val}</div><div class='lbl'>{lbl}</div></div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_chat, tab_cases, tab_eval, tab_bench = st.tabs(["💬 Legal Chat", "⚖️ Case Suite", "📊 Evaluation", "🧠 Model Benchmarks"])


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — CHAT
# ═══════════════════════════════════════════════════════════════════════════════
with tab_chat:
    if not st.session_state.indexed:
        st.info("⚡ Click **Index Legal Documents** in the sidebar to get started.")
    else:
        col_chat, col_steps = st.columns([2, 1])

        with col_chat:
            st.markdown("### 💬 Ask a Legal Question")
            chat_box = st.container()
            with chat_box:
                for msg in st.session_state.chat:
                    if msg["role"] == "user":
                        st.markdown(f"<div class='user-msg'>🙋 {msg['content']}</div>", unsafe_allow_html=True)
                    else:
                        src_html = "".join(f"<span class='src-badge'>📄 {s}</span>" for s in msg.get("sources", []))
                        st.markdown(f"<div class='ai-msg'>🧠 {msg['content']}"
                                    + (f"<br><br>{src_html}" if src_html else "") + "</div>",
                                    unsafe_allow_html=True)

            with st.form("chat_form", clear_on_submit=True):
                q = st.text_input("Your legal question…", placeholder="e.g. Is COVID-19 a valid force majeure event?", label_visibility="collapsed")
                submitted = st.form_submit_button("Send ➤")

            if submitted and q.strip():
                _run_query(q.strip(), selected_model, openai_key, gemini_key, deepseek_key)
                st.rerun()

        with col_steps:
            st.markdown("### 🔍 Agent Reasoning Trace")
            if st.session_state.chat:
                last_steps = st.session_state.chat[-1].get("steps", []) if st.session_state.chat[-1]["role"] == "assistant" else []
                if last_steps:
                    for step in last_steps:
                        st.markdown(f"<div class='step-item'>{step}</div>", unsafe_allow_html=True)
                else:
                    st.caption("Steps from last query will appear here.")
            else:
                st.caption("Agent reasoning steps appear here after each query.")


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — CASE SUITE
# ═══════════════════════════════════════════════════════════════════════════════
with tab_cases:
    from cases import LEGAL_CASES

    st.markdown("### ⚖️ Capstone Case Test Suite")
    st.caption("10 pre-loaded realistic legal scenarios. Click **Run** to execute against the RAG agent.")

    tag_map = {"case_law": ("Case Law","tag-cl"), "contracts": ("Contract","tag-ct"), "both": ("Both","tag-bt")}

    for case in LEGAL_CASES:
        cat_label, cat_cls = tag_map.get(case["category"], ("General","tag-cl"))
        with st.expander(f"⚖️ Case {case['id']}: {case['title']}"):
            st.markdown(f"<span class='tag {cat_cls}'>{cat_label}</span>", unsafe_allow_html=True)
            st.markdown(f"**Facts:** {case['facts']}")
            st.markdown("**Legal Issues:**")
            for issue in case["legal_issues"]:
                st.markdown(f"- {issue}")
            st.markdown(f"**RAG Query:** `{case['rag_query']}`")

            if st.session_state.indexed:
                if st.button(f"▶ Run Case {case['id']}", key=f"run_{case['id']}"):
                    _run_query(case["rag_query"], selected_model, openai_key, gemini_key, deepseek_key)
                    st.session_state["last_case"] = case["id"]
                    st.rerun()
            else:
                st.warning("Index documents first.")


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════
with tab_eval:
    import pandas as pd, plotly.express as px, plotly.graph_objects as go

    st.markdown("### 📊 Evaluation Metrics")

    if not st.session_state.eval_log:
        st.info("No queries evaluated yet. Ask a question in the Chat tab.")
    else:
        df = pd.DataFrame(st.session_state.eval_log)

        m1, m2, m3, m4 = st.columns(4)
        with m1: st.metric("Avg Faithfulness", f"{df['faithfulness'].mean():.2%}")
        with m2: st.metric("Avg Latency",      f"{df['latency_s'].mean():.2f}s")
        with m3: st.metric("Total Cost",        f"${df['cost_usd'].sum():.5f}")
        with m4: st.metric("Total Queries",     len(df))

        st.markdown("<br>", unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            fig = px.line(df, x=df.index, y="faithfulness", title="Faithfulness Score per Query",
                         labels={"x":"Query #","faithfulness":"Score"},
                         color_discrete_sequence=["#667eea"])
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.03)",
                              font_color="#e2e8f0", title_font_size=14)
            fig.add_hline(y=0.8, line_dash="dot", line_color="#f093fb", annotation_text="Target: 0.8")
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            fig2 = px.bar(df, x=df.index, y="latency_s", title="Latency per Query (seconds)",
                          color="latency_s", color_continuous_scale=["#667eea","#f093fb"],
                          labels={"x":"Query #","latency_s":"Seconds"})
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.03)",
                               font_color="#e2e8f0", title_font_size=14, showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("#### Query Log")
        display_cols = ["query","faithfulness","latency_s","cost_usd","model"]
        available = [c for c in display_cols if c in df.columns]
        st.dataframe(df[available].style.format({"faithfulness":"{:.2%}","latency_s":"{:.2f}s","cost_usd":"${:.6f}"}),
                     use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — BENCHMARKS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_bench:
    import plotly.graph_objects as go
    st.markdown("### 🧠 Model Selection Analysis")

    # HELM LegalBench
    st.markdown("#### Stanford HELM — LegalBench Scores")
    helm_data = {
        "Model": ["GPT-4o","Gemini 1.5 Pro","GPT-4o-mini","DeepSeek-V3","Mistral-Large","Claude 3.5 Sonnet"],
        "Exact Match (%)": [72.3, 68.9, 61.4, 66.8, 59.2, 70.1],
        "F1 Score":        [0.81, 0.77, 0.69, 0.74, 0.65, 0.79],
        "BoolQ (%)":       [88.4, 85.1, 82.3, 83.7, 79.8, 87.2],
    }
    df_helm = pd.DataFrame(helm_data)
    fig_helm = px.bar(df_helm, x="Model", y="Exact Match (%)", color="Model",
                      color_discrete_sequence=px.colors.sequential.Plasma,
                      title="LegalBench Exact Match — Higher is Better")
    fig_helm.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.03)",
                           font_color="#e2e8f0", showlegend=False)
    st.plotly_chart(fig_helm, use_container_width=True)
    st.dataframe(df_helm.set_index("Model"), use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Artificial Analysis
    st.markdown("#### Artificial Analysis — Cost vs Quality")
    aa_data = {
        "Model": ["GPT-4o","Gemini 1.5 Pro","GPT-4o-mini","DeepSeek-V3","Gemini Flash"],
        "Quality Index": [85, 82, 71, 78, 69],
        "Cost $/1M input": [5.00, 3.50, 0.15, 0.27, 0.075],
        "Speed (tok/s)":  [110, 95, 180, 200, 250],
        "Context (K)":    [128, 1000, 128, 64, 1000],
    }
    df_aa = pd.DataFrame(aa_data)
    fig_aa = px.scatter(df_aa, x="Cost $/1M input", y="Quality Index",
                        size="Speed (tok/s)", color="Model", text="Model",
                        title="Cost vs Quality (bubble = speed)",
                        color_discrete_sequence=px.colors.qualitative.Vivid)
    fig_aa.update_traces(textposition="top center")
    fig_aa.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.03)",
                         font_color="#e2e8f0")
    st.plotly_chart(fig_aa, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # LM Arena Elo
    st.markdown("#### LM Chat Arena — Elo Rankings (Instruction Following)")
    elo_data = {
        "Model":["GPT-4o","Claude 3.5 Sonnet","Gemini 1.5 Pro","GPT-4o-mini","DeepSeek-V3","Mistral-Large"],
        "Elo Score":[1286,1272,1263,1240,1255,1208],
    }
    df_elo = pd.DataFrame(elo_data).sort_values("Elo Score", ascending=True)
    fig_elo = go.Figure(go.Bar(x=df_elo["Elo Score"], y=df_elo["Model"], orientation="h",
                               marker_color=["#667eea","#764ba2","#f093fb","#a5b4fc","#818cf8","#c4b5fd"]))
    fig_elo.update_layout(title="Chatbot Arena Elo (higher = better)", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(255,255,255,0.03)", font_color="#e2e8f0",
                          xaxis_range=[1180, 1310])
    st.plotly_chart(fig_elo, use_container_width=True)

    st.markdown("""
    > **My Recommendation for Legal AI:**
    > - **Primary:** GPT-4o — best legal reasoning, top HELM scores, strong citation accuracy
    > - **Long Context:** Gemini 1.5 Pro — 1M token window handles full contracts
    > - **Cost-Efficient:** DeepSeek-V3 — 95% quality at 5% the cost for high-volume tasks
    """)

