# -------------------------------
# 🏭 Jaggery Batch Assistant (Fixed Timestamp Version)
# Ollama + TimescaleDB RAG Script for Streamlit
# -------------------------------

import streamlit as st
import psycopg2
import pandas as pd
import requests
import json
from datetime import datetime
import sys

print(sys.executable)  # Debug: Confirm Python path

# -------------------------------
# Constants
# -------------------------------

MODEL_NAME = "granite3.3:8b"
OLLAMA_HOST = "http://localhost:11434"

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

Your task is to convert the user's question into a safe and syntactically correct SQL SELECT query for PostgreSQL.
- Only generate the SQL query. No explanations or markdown.
- Use timestamp comparisons correctly (BETWEEN, >=, <=).
- ONLY output SELECT queries. Never use UPDATE, DELETE, DROP, etc.

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
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="🏭 Jaggery Batch Assistant", layout="centered")
st.title("🏭 Jaggery Batch Assistant")
st.caption(f"Conversational interface for TimescaleDB, powered by Ollama ({MODEL_NAME}).")

# -------------------------------
# Database Connection
# -------------------------------

def init_connection():
    """Initialize connection to TimescaleDB."""
    try:
        st.write("🔌 Connecting to PostgreSQL...")
        conn = psycopg2.connect(
            dbname="postgres",
            user="postgres",
            password="aura",  # change to your password
            host="localhost",
            port=5439
        )
        st.success("✅ Database connected successfully")
        return conn
    except Exception as e:
        st.error(f"❌ Failed to connect to DB: {e}")
        st.stop()

conn = init_connection()

# -------------------------------
# Helper: Timestamp Cleaner
# -------------------------------

def convert_timestamps(df):
    """Convert numeric or epoch-like timestamps to human-readable format."""
    for col in ["start_time", "end_time", "next_batch_start"]:
        if col in df.columns:
            try:
                # Try to detect if timestamps are numeric epoch
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = pd.to_datetime(df[col], unit='ms', errors='coerce')
                else:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
            except Exception as e:
                print(f"Timestamp conversion failed for {col}: {e}")
    return df

# -------------------------------
# Run SQL Query
# -------------------------------

def run_query(sql_query):
    """Execute SQL safely and return cleaned DataFrame."""
    try:
        df = pd.read_sql(sql_query, conn)
        df = convert_timestamps(df)  # Convert all timestamps properly
        return df
    except Exception as e:
        return pd.DataFrame([{'error': str(e)}])

# -------------------------------
# Ask Ollama
# -------------------------------
@st.cache_data(show_spinner=False)
def ask_ollama(full_prompt, model_name=MODEL_NAME):
    """Ask Ollama via REST API."""
    url = f"{OLLAMA_HOST}/api/generate"
    payload = {"model": model_name, "prompt": full_prompt, "stream": False}
    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()
    except requests.exceptions.ConnectionError:
        st.error(f"⚠️ Could not connect to Ollama at {OLLAMA_HOST}. Is it running?")
        return None
    except Exception as e:
        st.error(f"Ollama API Error: {e}")
        return None

# -------------------------------
# Generate SQL Query from Natural Language
# -------------------------------

def generate_sql(question):
    """Convert user text into SQL query using Ollama."""
    prompt = SQL_GEN_PROMPT_TEMPLATE.format(question=question)
    sql_query = ask_ollama(prompt)

    if not sql_query:
        return None

    sql_query = sql_query.strip().strip("`")
    if sql_query.lower().startswith("sql"):
        sql_query = sql_query[3:].strip()

    if not sql_query.lstrip().upper().startswith("SELECT"):
        st.error("❌ Unsafe SQL generated (not SELECT). Try rephrasing.")
        return None

    return sql_query

# -------------------------------
# RAG Pipeline
# -------------------------------

def answer_question(question):
    """Main flow: Generate SQL → Fetch Data → Ask Ollama."""
    sql_query = generate_sql(question)
    if not sql_query:
        return "I couldn’t generate a valid SQL query. Please rephrase your question."

    with st.expander("🔍 View Generated SQL Query"):
        st.code(sql_query, language="sql")

    df = run_query(sql_query)

    if "error" in df.columns:
        error_message = df["error"].iloc[0]
        st.error(f"SQL Error: {error_message}")
        results = {"error": error_message}
    else:
        if df.empty:
            results = "No data found."
        else:
            results = df.to_markdown(index=False)

    rag_prompt = RAG_PROMPT_TEMPLATE.format(results=results, question=question)
    response = ask_ollama(rag_prompt)

    if response is None:
        response = "I retrieved the data, but encountered an issue analyzing it with the LLM."

    return response

# -------------------------------
# Chat Interface
# -------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "👋 Hi sir! I’m your Jaggery Batch Assistant. Ask me about batch times, delays, or status."
    }]

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
if prompt := st.chat_input("Ask about batch data..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = answer_question(prompt)
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
