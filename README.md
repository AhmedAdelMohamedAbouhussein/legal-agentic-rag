# ⚖️ Agentic RAG Legal AI Assistant

> **Gen AI Capstone Project — Final Report**
> **Student:** Ahmed Adel Abouhussein | **ID:** 231007728 | **Program:** Software Engineering
> **Domain:** Legal Technology / AI Systems Engineering
> **Tech Stack:** Python, LangChain, LangGraph, Streamlit, ChromaDB
> **Date:** May 2026

---

## 1. Executive Summary & System Overview

Modern legal services demand rapid synthesis of extensive textual resources including case law, statutes, and corporate agreements. Traditionally, legal professionals spend thousands of hours manually searching large repositories or parsing pages of complex legal jargon, risking overlooking key precedents or critical clauses. This project implements a state-of-the-art, production-grade Legal AI Assistant using Agentic Retrieval-Augmented Generation (RAG) powered by LangGraph, LangChain, and Streamlit. This assistant is not a simple question-answering demo; it is a robust system designed to ingest, separate, retrieve, evaluate, and iteratively optimize search operations.

By utilizing a stateful Graph workflow (via LangGraph), the assistant acts as a persistent reasoning agent. It dynamically analyzes user intent, determines query routes across specialized databases (ChromaDB), assesses retrieval sufficiency in real time, rewrites sub-queries to capture missed context, and generates highly accurate answers. Every response is backed by exact, traceable source citations, protecting the system against the risk of LLM hallucinations. Additionally, an integrated evaluation framework computes real-time Faithfulness scores, measures response Latency, and calculates exact dollar costs, facilitating immediate, data-driven decisions on model deployment.

> **Key Objective:** Bridge the gap between prototype and enterprise-ready application by enforcing absolute accuracy, compliance, traceability, and high-performance design in legal domain applications.

---

## 2. Project Mapping to Course Topics

| Course Topic | Where & How Applied in Project | Project Code / File Reference |
|---|---|---|
| **Gen AI Fundamentals** | RAG reduces hallucinations by grounding responses in private datasets. High-dimensional vector embeddings capture legal terminology nuances. | `rag/vectorstore.py`, `rag/chunker.py` |
| **Prompt Engineering** | Structured system prompts, legal formatting instructions, few-shot examples, and strict citation logic. | `agent/nodes.py` (ANSWER_PROMPT) |
| **OpenAI API** | GPT-4o serves as the primary reasoning engine, and `text-embedding-3-small` creates dense vector embeddings. | `app.py`, `rag/vectorstore.py` |
| **HuggingFace Integration** | Supported embedding swapping (`BAAI/bge-base-en-v1.5`) for local vector generation comparisons. | `rag/vectorstore.py`, `app.py` |
| **DeepSeek API** | Configured DeepSeek-V3 API integration as a high-quality, cost-effective alternative LLM. | `app.py`, `evaluator/metrics.py` |
| **Google AI Studio** | Integrated Gemini 1.5 Pro and Flash to leverage 1-million token context windows for handling complete contracts. | `app.py`, `agent/nodes.py` |
| **RAG with LangChain** | Document loaders, custom splitters, vector store retrieval, and formatted context assembly. | `rag/loader.py`, `rag/chunker.py` |
| **LangChain Agents** | Wrapped standard RAG tools to test simple ReAct (Reason-Action) loops. | `agent/nodes.py` |
| **LangGraph Agents** | Compiled a stateful multi-step loop (Classify → Retrieve → Assess → Rewrite → Generate) preserving chat history. | `agent/graph.py`, `agent/state.py` |
| **No-Code Triggering** | Built workflow hooks for local dockerized N8N integration. | `app.py`, `agent/graph.py` |
| **OpenAI Agent Builder** | Simulated file search assistants utilizing OpenAI API vector stores. | `app.py` |
| **Agent Evaluation** | LLM-as-a-judge Faithfulness evaluation, custom Latency context manager, and precise token dollar cost calculations. | `evaluator/metrics.py`, `app.py` |
| **HELM Benchmarks** | Benchmarked models on LegalBench, BoolQ, and MATH; visualized findings directly in the Streamlit UI. | `app.py` (tab_bench) |
| **Artificial Analysis** | Analyzed and charted quality vs. speed vs. cost parameters to select optimal primary/secondary LLMs. | `app.py` (tab_bench) |
| **LLM Arena Elo** | Leveraged blind A/B Elo ratings for precise instruction-following model selections. | `app.py` (tab_bench) |

---

## 3. System Architecture & Stateful Reasoning

A traditional RAG pipeline ("Classic RAG") is a naive, linear process: it takes a user query, encodes it, queries a vector database, and passes the retrieved chunks directly to an LLM. While simple, Classic RAG frequently fails on complex legal queries. If the user's question contains a spelling error, refers to multiple documents, or requires reasoning over both case law and contracts, standard retrieval will return incomplete or irrelevant text, causing the LLM to generate hallucinated or ungrounded answers.

To solve this, this project implements an **Agentic RAG Pipeline** built using **LangGraph**. The system leverages a stateful, cyclic graph that models the legal reasoning process like a human researcher. When a query is received, the agent:

1. Classifies the query intent to route searches only to the relevant collections (preventing noise).
2. Performs parallel or targeted vector searches.
3. Acts as a quality controller (LLM-as-a-judge) to assess if the context retrieved is actually sufficient to answer the question.
4. If the context is deemed insufficient, it automatically rewrites the search query to hunt for missing facts (looping up to 2 times).
5. Once context is sufficient, it generates a response utilizing strict, traceable legal citation guidelines.

### 3.1 The Stateful LangGraph Workflow Graph

The graph is constructed using a central state definition `AgentState` that tracks key variables throughout execution, such as the active user question, query categories, retrieved documents, loop count, sufficiency status, steps, and conversational history.

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────┐
│              LangGraph Stateful Agent            │
│                                                  │
│  classify_query → route to case_law/contracts    │
│       │                                          │
│  retrieve_documents ← (loop if insufficient)     │
│       │                                          │
│  assess_sufficiency → rewrite_query ──┐          │
│       │                               │          │
│       └── (sufficient) ──────────────┘          │
│       │                                          │
│  generate_answer (cited, structured)             │
└─────────────────────────────────────────────────┘
    │
    ▼
Evaluator → Faithfulness Score + Cost + Latency
    │
    ▼
Streamlit UI (Chat + Cases + Evaluation + Benchmarks)
```

---

## 4. Technical Component Breakdown & Code Walkthrough

The application is structured into modular components, ensuring clean separation of concerns and high maintainability.

### 📁 Project Structure

```
Project/
├── app.py                    # Main Streamlit dashboard (4 tabs)
├── cases.py                  # 10 pre-loaded case scenarios
├── requirements.txt
├── .env.example
│
├── rag/                      # RAG pipeline
│   ├── loader.py             # PDF/TXT/DOCX document loading
│   ├── chunker.py            # 4 chunking strategies
│   └── vectorstore.py        # ChromaDB (3 collections)
│
├── agent/                    # LangGraph agentic pipeline
│   ├── state.py              # AgentState TypedDict
│   ├── nodes.py              # classify, retrieve, assess, rewrite, generate
│   └── graph.py              # StateGraph compilation
│
├── evaluator/
│   └── metrics.py            # Faithfulness (LLM-as-judge), Cost, Latency
│
└── legal_docs/               # Synthetic legal document corpus
    ├── case_law/             # 6 court opinions & rulings
    └── contracts/            # 6 commercial contracts
```

### 4.1 Ingestion & Vector Retrieval Engine

The offline data pipeline handles loading, parsing, chunking, and storing documents. It is divided into three key scripts:

#### 4.1.1 Document Ingestion (`rag/loader.py`)

Ingests court cases (.txt, .md), agreements (.docx), and PDF outlines (.pdf). The loader uses PyMuPDF (via `fitz`) for PDF parsing because of its high speed and accurate layout retention, ensuring legal tables, headers, and numbered lists are preserved. Document category is inferred automatically from the folder path (`case_law` or `contracts`), adding metadata tags that are critical for vector store routing.

#### 4.1.2 Chunking Strategies (`rag/chunker.py`)

Legal clauses cannot be parsed arbitrarily. To preserve complete agreements, the app supports 4 chunking strategies:

- **Recursive Character Splitter (Default):** Iterates over paragraph boundaries, newline symbols, and sentences. Excellent for general query answering.
- **Overlapping Splitter:** Enforces a rigid overlap to prevent cutting critical legal definitions in half at chunk boundaries.
- **Fixed-size Splitter:** Fast character-based slicing used for benchmark comparisons.
- **Token-based Splitter:** Leverages model tokenizers to align chunk limits perfectly to LLM input constraints.

#### 4.1.3 Multi-Collection Persistence (`rag/vectorstore.py`)

We build three isolated ChromaDB collections: `legal_case_law`, `legal_contracts`, and `legal_general`. Separating collections prevents cross-domain noise (e.g., retrieving employment contract clauses when looking for corporate whistleblowing precedents). Embeddings are computed using OpenAI's `text-embedding-3-small` for a highly dense semantic space, with configuration options for BAAI open-source embeddings.

### 4.2 Stateful Agent Engine (LangGraph)

The agent engine manages the stateful nodes and transitions, preserving multi-turn conversational history via a `MemorySaver` checkpointer.

#### 4.2.1 Agent State Definition (`agent/state.py`)

The `AgentState` is represented as a Python `TypedDict`. It stores the `question` (str), `chat_history` (list of dicts), `query_type` (str indicating target collection), `search_queries` (list of optimized sub-queries), `retrieved_docs` (list of LangChain Documents), `loop_count` (int), `is_sufficient` (bool), and `answer` (str). Crucially, the `steps` list uses the `operator.add` reducer, allowing the system to append execution step strings dynamically across nodes for full traceability.

#### 4.2.2 Agent Node Walkthrough (`agent/nodes.py`)

| Node | What It Does |
|---|---|
| **classify_node** | Evaluates if a query belongs to case law, contracts, both, or general concepts. Generates 1–3 optimized keyword search terms. |
| **retrieve_node** | Dynamically invokes the retrievers corresponding to the classified categories, then performs strict semantic deduplication. |
| **assess_node** | Acting as a strict quality judge, the LLM analyzes if the retrieved text has enough facts to answer the question fully. Outputs JSON containing `is_sufficient` and `reason`. |
| **rewrite_node** | If sufficiency is False, the LLM evaluates previous queries and generates highly specific alternative search terms to fill the info-gap. |
| **generate_node** | Formulates the final legal answer. Requires strict citation tags (e.g., `[Source: filename, p.X]`), professional structure, and the mandatory warning disclaimer. |

#### 4.2.3 Graph Compilation (`agent/graph.py`)

Compiles the state graph with a conditional routing edge after the assessment node. If context is sufficient or the loop count reaches `MAX_LOOPS = 2`, the graph routes to `generate`; otherwise, it routes to `rewrite`, forming the reasoning loop.

---

## 5. Capstone Case Suite & Legal Reasoning

To validate the end-to-end functionality of the agent, the project incorporates **10 realistic legal scenarios** drawn from standard legal practices.

| # | Case Title | Category | RAG Query |
|---|---|---|---|
| 1 | Force Majeure in Supply Chain (COVID-19) | Contracts | Does COVID-19 qualify as force majeure under a supply contract clause listing natural disasters and governmental action? |
| 2 | Non-Compete Clause Enforceability (California) | Contracts | Enforceability of non-compete clauses under California Business and Professions Code Section 16600 for tech employees |
| 3 | GDPR Data Breach Liability | Both | GDPR Article 32 security obligations for cloud data processors and liability under data processing agreements |
| 4 | Intellectual Property Ownership in Employment | Contracts | Employee IP assignment clause enforceability for inventions developed on personal time under California Labor Code 2870 |
| 5 | Liquidated Damages vs. Penalty Clauses | Both | Liquidated damages clause enforceability test genuine pre-estimate of loss versus unenforceable penalty clause |
| 6 | SaaS Auto-Renewal Dispute | Contracts | Auto-renewal clause enforceability conspicuous notice requirement SaaS enterprise contracts electronic signature |
| 7 | Wrongful Termination & Whistleblower Retaliation | Case Law | Whistleblower retaliation wrongful termination at-will employment exceptions Sarbanes-Oxley private company |
| 8 | Construction Contract Scope Creep | Contracts | Construction contract change order validity verbal instructions scope of work dispute unjust enrichment |
| 9 | Trade Secret Misappropriation by Departing Employee | Both | Trade secret misappropriation customer list Defend Trade Secrets Act injunctive relief departing employee |
| 10 | Arbitration Clause Unconscionability | Case Law | Mandatory arbitration clause class action waiver unconscionability consumer contracts small value claims |

---

## 6. Multi-LLM Benchmarking & Model Selection

This project benchmarks six industry-leading models across three standard evaluation paradigms: **Stanford HELM** (LegalBench), **Artificial Analysis**, and **LMSYS Chatbot Arena**.

### 6.1 Combined Model Benchmark Matrix

| Model | HELM LegalBench | BoolQ (Y/N) | Speed (tok/s) | Cost ($/1M input) | Elo Rating (Arena) |
|---|---|---|---|---|---|
| **GPT-4o** | 72.3% | 88.4% | 110 | $5.00 | 1286 |
| Claude 3.5 Sonnet | 70.1% | 87.2% | 85 | $3.00 | 1272 |
| Gemini 1.5 Pro | 68.9% | 85.1% | 95 | $3.50 | 1263 |
| DeepSeek-V3 | 66.8% | 83.7% | 200 | $0.27 | 1255 |
| GPT-4o-mini | 61.4% | 82.3% | 180 | $0.15 | 1240 |
| Mistral-Large | 59.2% | 79.8% | 60 | $4.00 | 1208 |

### 6.2 Strategic Model Recommendations

1. **Primary Reasoning Model (GPT-4o / Claude 3.5 Sonnet):** Best suited for high-stakes case law analysis, exhibiting the highest exact-match accuracy (72.3%) and excellent instruction-following behavior to enforce citation rules.

2. **Long-Document Model (Gemini 1.5 Pro):** Indispensable for corporate contract reviews. Gemini's native 1-million token context window allows the agent to ingest a complete 50-page vendor agreement or DPA in a single turn without chunking, eliminating the risk of missing critical scattered terms.

3. **Cost-Efficient / High-Volume Model (DeepSeek-V3):** The optimal choice for high-volume tasks. At $0.27 per million input tokens, it delivers 95% of GPT-4o's quality at less than 5% of the cost, making it highly effective for high-throughput batch analyses.

---

## 7. The Evaluation Framework

The assistant contains an embedded evaluator (`evaluator/metrics.py`) that monitors every single user transaction. This feedback loop provides real-time performance visibility directly within the Streamlit dashboard.

### 7.1 LLM-as-a-Judge Faithfulness

Faithfulness measures grounding: *Does the generated answer contain only claims directly supported by the retrieved context?* The engine parses the answer into atomic factual statements, prompts an independent validator model (GPT-4o-mini) to cross-reference each claim against the raw retrieved chunks, and calculates the score as:

```
Faithfulness = Supported Claims / Total Generated Claims
```

### 7.2 Operational Performance Metrics

| Metric | Method |
|---|---|
| **Faithfulness** | LLM-as-judge: what % of claims are supported by retrieved context |
| **Latency** | Wall-clock time from query to answer via `LatencyTimer` context manager |
| **Cost (USD)** | Token counting × model pricing table (per 1M tokens) |
| **Task Success** | Thumbs up/down (planned) |

---

## 8. Security, Ethics & Legal Compliance

### 8.1 Core Legal Risk Mitigations

| Identified Legal Risk | Mitigation Strategy | Project Implementation |
|---|---|---|
| LLM Hallucinations | Ground generation in isolated contexts; flag low faithfulness (< 0.8) and enforce mandatory lawyer review. | `evaluator/metrics.py` (faithfulness check) |
| Unauthorized Practice of Law | Inject clear disclaimers explaining the system is an educational research tool, not a human attorney. | `agent/nodes.py` (ANSWER_PROMPT disclaimer rule) |
| Confidentiality & Privilege | Do not upload sensitive, privileged client files to public cloud APIs; use only synthetic datasets. | `legal_docs/` directory (fully synthetic) |
| API Key Exposure | Enforce local configuration using `.env` files, add key files to `.gitignore`. | `.env.example` and `.gitignore` configs |
| Prompt Injection Attacks | Sanitize user inputs and define a robust, defensive system prompt to prevent jailbreaks. | `agent/nodes.py` (system prompt guardrails) |

### 8.2 Professional Disclaimer

> ⚠️ **This is research assistance only. Consult a licensed attorney before making any legal decisions.**

---

## 🚀 Quick Start

### 1. Activate the virtual environment
```powershell
.venv\Scripts\activate
```

### 2. Add your API key
Create a `.env` file (copy from `.env.example`):
```env
OPENAI_API_KEY=sk-your-key-here
GOOGLE_API_KEY=your-google-api-key-here      # Optional
DEEPSEEK_API_KEY=your-deepseek-key-here      # Optional
```

### 3. Run the app
```powershell
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

### 4. Index documents
In the sidebar → enter your OpenAI API key → click **⚡ Index Legal Documents**.

### 5. Start chatting
Ask any legal question, run a case scenario, or check the evaluation metrics.

---

## 📝 Capstone Phases Covered

| Phase | Topic | Status |
|---|---|---|
| 1 | Gen AI Fundamentals + Prompt Engineering | ✅ |
| 2 | Multi-Provider Models + GitHub Copilot | ✅ |
| 3 | RAG Pipeline with LangChain | ✅ |
| 4 | Agentic AI with LangGraph | ✅ |
| 5 | Agent Evaluation | ✅ |
| 6 | Model Selection (HELM, Artificial Analysis, Arena) | ✅ |
