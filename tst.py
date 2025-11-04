# app.py
# -------------------------------
# 🏭 Jaggery Batch Assistant — Full Refactor
# Streamlit + TimescaleDB (SQLAlchemy) + Ollama RAG
# Auto visualizations with matplotlib (no DB writes)
# -------------------------------

import streamlit as st
from sqlalchemy import create_engine, text
import pandas as pd
import requests
import json
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import re

# -------------------------------
# Configuration — change as needed
# -------------------------------
MODEL_NAME = "granite3.3:8b"
OLLAMA_HOST = "http://localhost:11434"
DB_USER = "postgres"
DB_PASS = "aura"
DB_NAME = "postgres"
DB_HOST = "localhost"
DB_PORT = 5439
DB_URI = f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# SQL generation prompt (strictly SELECT)
DB_SCHEMA_INFO = """
This table has the following columns:
batch_id (integer),
start_time (timestamp or bigint epoch),
end_time (timestamp or bigint epoch),
next_batch_start (timestamp or bigint epoch),
gap_delay (string: 'normal', 'failure', 'perfect', or 'cleaning_delay').
"""

SQL_GEN_PROMPT_TEMPLATE = f"""
You are an expert data assistant with access to a PostgreSQL table named 'batch_data'.
{DB_SCHEMA_INFO}

Only produce a single safe SQL SELECT query (PostgreSQL syntax) that answers the user's question.
- Output must be only the SQL query, nothing else.
- Use SELECT ... FROM batch_data ...
- Use WHERE and timestamp comparisons correctly (BETWEEN, >=, <=) when needed.
- NEVER output UPDATE, DELETE, INSERT, DROP, TRUNCATE, or any DDL/DML.
User Question: "{{question}}"
SQL Query:
"""

RAG_PROMPT_TEMPLATE = """
You are a helpful assistant for the Jaggery Batch production team.
Based on the following database data, answer the user's question clearly.
If there is no data, say so. If there's an error, describe it briefly.

Database Results:
{results}

User Question:
"{question}"

Answer:
"""

# -------------------------------
# Streamlit UI setup
# -------------------------------
st.set_page_config(page_title="🏭 Jaggery Batch Assistant", layout="centered")
st.title("🏭 Jaggery Batch Assistant")
st.caption("Conversational interface for TimescaleDB, powered by Ollama.")

# Show Python path for debugging (optional)
st.text(f"Python: {sys.executable}")

# -------------------------------
# Database connection (SQLAlchemy)
# -------------------------------
@st.cache_resource
def init_engine():
    engine = create_engine(DB_URI, pool_pre_ping=True)
    return engine

engine = init_engine()

# -------------------------------
# Helper: Ollama call
# -------------------------------
@st.cache_data(show_spinner=False)
def ask_ollama(full_prompt, model_name=MODEL_NAME, timeout=60):
    url = f"{OLLAMA_HOST}/api/generate"
    payload = {"model": model_name, "prompt": full_prompt, "stream": False}
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        return data.get("response", "").strip()
    except requests.exceptions.ConnectionError:
        st.error(f"⚠️ Could not connect to Ollama at {OLLAMA_HOST}. Is it running?")
        return None
    except Exception as e:
        st.error(f"Ollama API Error: {e}")
        return None

# -------------------------------
# Helper: sanitize SQL (ensure SELECT only)
# -------------------------------
def sanitize_sql(sql_text):
    if not sql_text:
        return None
    # remove markdown/backticks
    sql = sql_text.strip().strip("` ").splitlines()
    sql = " ".join([line.strip() for line in sql if line.strip()])
    # basic safety: only allow SELECT ... FROM ...
    if re.match(r"(?i)^\s*select\s+", sql) and not re.search(r"(?i)\b(update|delete|insert|drop|truncate|alter|create)\b", sql):
        return sql
    return None

# -------------------------------
# Timestamp conversion helper
# -------------------------------
def convert_timestamps(df):
    for col in ["start_time", "end_time", "next_batch_start"]:
        if col in df.columns:
            try:
                if pd.api.types.is_numeric_dtype(df[col]):
                    # decide ms vs seconds by magnitude
                    sample = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
                    if sample is not None and sample > 1e12:
                        df[col] = pd.to_datetime(df[col], unit="ms", errors="coerce")
                    else:
                        df[col] = pd.to_datetime(df[col], unit="s", errors="coerce")
                else:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
            except Exception:
                df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

# -------------------------------
# Run SQL safely (read-only)
# -------------------------------
def run_select_query(sql_query):
    try:
        with engine.connect() as conn:
            df = pd.read_sql(text(sql_query), conn)
        df = convert_timestamps(df)
        return df
    except Exception as e:
        return pd.DataFrame([{"error": str(e)}])

# -------------------------------
# Analysis & Visualization functions (matplotlib only)
# -------------------------------
def plot_timeseries(df, x_col, y_cols, title=None):
    fig, ax = plt.subplots()
    for col in y_cols:
        if col in df.columns:
            ax.plot(df[x_col], df[col], marker="o", label=str(col))
    ax.set_xlabel(x_col)
    ax.set_ylabel(", ".join(map(str, y_cols)))
    if title:
        ax.set_title(title)
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

def plot_histogram(df, col, bins=20, title=None):
    fig, ax = plt.subplots()
    ax.hist(df[col].dropna(), bins=bins)
    ax.set_xlabel(col)
    ax.set_ylabel("Frequency")
    if title:
        ax.set_title(title)
    ax.grid(True)
    st.pyplot(fig)

def plot_scatter(df, x_col, y_col, title=None):
    fig, ax = plt.subplots()
    ax.scatter(df[x_col], df[y_col])
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    if title:
        ax.set_title(title)
    ax.grid(True)
    st.pyplot(fig)

def plot_bar(summary_df, x_col, y_col, title=None):
    fig, ax = plt.subplots()
    ax.bar(summary_df[x_col], summary_df[y_col])
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    if title:
        ax.set_title(title)
    ax.grid(True)
    st.pyplot(fig)

def plot_correlation_heatmap(df):
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] < 2:
        st.info("Not enough numeric columns for correlation.")
        return
    corr = numeric.corr()
    fig, ax = plt.subplots()
    cax = ax.matshow(corr, vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = range(len(corr.columns))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(corr.columns, rotation=45, ha="left")
    ax.set_yticklabels(corr.columns)
    ax.set_title("Correlation matrix")
    st.pyplot(fig)
    st.dataframe(corr)

# Gap analysis
def analyze_gaps(df):
    # require start_time and end_time
    if not all(c in df.columns for c in ["start_time", "end_time", "batch_id"]):
        st.error("Gap analysis requires batch_id, start_time and end_time in the results.")
        return None
    d = df.sort_values(["batch_id"]).reset_index(drop=True).copy()
    d["gap_in_seconds"] = (d["start_time"] - d["end_time"].shift(1)).dt.total_seconds().fillna(0)
    def classify(g):
        if pd.isna(g):
            return "unknown"
        if g >= 300:
            return "cleaning_delay"
        if g == 0:
            return "failure"
        if g >= 30:
            return "normal"
        if 1 <= g < 30:
            return "perfect"
        return "unknown"
    d["gap_delay"] = d["gap_in_seconds"].apply(classify)
    summary = d.groupby("gap_delay", as_index=False)["gap_in_seconds"].mean().sort_values("gap_in_seconds", ascending=False)
    st.subheader("🧾 Average Gap by Delay Type")
    st.dataframe(summary.rename(columns={"gap_in_seconds": "avg_gap_seconds"}))
    plot_bar(summary, "gap_delay", "gap_in_seconds", title="Average gap by type (seconds)")
    return d

# -------------------------------
# Intent detection (rudimentary but effective)
# -------------------------------
VIS_INTENT_KEYWORDS = ["plot", "visualize", "show chart", "graph", "hist", "scatter", "time series", "correlation", "correlate", "gap", "difference", "compare"]

def detect_visual_intent(question):
    q = question.lower()
    return any(k in q for k in VIS_INTENT_KEYWORDS)

# -------------------------------
# SQL generation wrapper
# -------------------------------
def generate_sql(question):
    prompt = SQL_GEN_PROMPT_TEMPLATE.format(question=question)
    sql_text = ask_ollama(prompt)
    sql = sanitize_sql(sql_text)
    if not sql:
        return None, sql_text
    return sql, sql_text

# -------------------------------
# Main RAG + Visualization pipeline
# -------------------------------
def answer_question(question):
    # 1) Generate SQL
    sql, raw_sql_response = generate_sql(question)
    if not sql:
        st.error("Could not generate a safe SELECT query from the question. Raw model output shown below for debugging.")
        st.code(raw_sql_response or "No response from Ollama.")
        return "Failed to generate safe SQL."

    with st.expander("🔍 Generated SQL (safe)"):
        st.code(sql, language="sql")

    # 2) Run SQL (read-only)
    df = run_select_query(sql)
    if "error" in df.columns:
        st.error(f"SQL execution error: {df['error'].iloc[0]}")
        return f"SQL error: {df['error'].iloc[0]}"

    if df.empty:
        st.info("Query returned no data.")
        return "No data found for that query."

    # show raw table (limit)
    st.subheader("📋 Query results (preview)")
    st.dataframe(df.head(200))

    # 3) If user asked for visualization or analysis, perform automatic visualizations
    if detect_visual_intent(question):
        st.info("Detected visualization/analysis intent — generating visualization(s).")

        # If the question includes 'gap' or 'difference', prefer gap analysis
        if "gap" in question.lower() or "difference" in question.lower():
            d = analyze_gaps(df)
            # also return short textual summary
            if d is not None:
                counts = d["gap_delay"].value_counts().to_dict()
                summary_text = "Gap classifications: " + ", ".join(f"{k}: {v}" for k, v in counts.items())
                st.write(summary_text)
                return "Visualization and gap analysis produced."

        # time series intent: look for timestamp and a numeric column to plot
        ts_cols = [c for c in df.columns if "time" in c.lower() or pd.api.types.is_datetime64_any_dtype(df[c])]
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if ts_cols and numeric_cols:
            x_col = ts_cols[0]
            # plot first 1-3 numeric columns
            y_cols = numeric_cols[:3]
            st.subheader("📈 Time series")
            plot_timeseries(df.sort_values(x_col), x_col, y_cols, title="Time series plot")
            return "Time series plotted."

        # correlation request
        if "correl" in question.lower():
            st.subheader("🔗 Correlation heatmap")
            plot_correlation_heatmap(df)
            return "Correlation plotted."

        # scatter request: try to detect two columns in question or pick top two numeric
        if "scatter" in question.lower():
            if len(numeric_cols) >= 2:
                plot_scatter(df, numeric_cols[0], numeric_cols[1], title=f"Scatter: {numeric_cols[0]} vs {numeric_cols[1]}")
                return "Scatter plotted."
            else:
                st.info("Not enough numeric columns for scatter plot.")
                return "Could not plot scatter (not enough numeric columns)."

        # histogram: if user asks for distribution or hist
        if "hist" in question.lower() or "distribution" in question.lower():
            if numeric_cols:
                plot_histogram(df, numeric_cols[0], bins=30, title=f"Distribution of {numeric_cols[0]}")
                return "Histogram plotted."
            else:
                st.info("No numeric columns to histogram.")
                return "Could not plot histogram."

        # fallback: if no specific instruction, try correlation + one histogram
        st.subheader("📊 Automatic summary visualizations")
        if numeric_cols:
            plot_histogram(df, numeric_cols[0], bins=20, title=f"Distribution: {numeric_cols[0]}")
        if len(numeric_cols) >= 2:
            plot_correlation_heatmap(df)
        return "Automatic visualizations generated."

    # 4) If no visualization intent, do RAG answer using Ollama
    # convert df to markdown for prompt (limit rows)
    md = df.head(200).to_markdown(index=False)
    rag_prompt = RAG_PROMPT_TEMPLATE.format(results=md, question=question)
    rag_answer = ask_ollama(rag_prompt)
    if rag_answer is None:
        return "Data retrieved successfully, but LLM failed to produce an analysis."
    return rag_answer

# -------------------------------
# Chat UI / Input handling
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "👋 Hi sir! I’m your Jaggery Batch Assistant. Ask me about batch times, visualize gaps, correlations, or request a table."
    }]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about batch data, visualizations, or analyses..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking & analyzing..."):
            answer = answer_question(prompt)
            st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})

# -------------------------------
# End of file
# -------------------------------
