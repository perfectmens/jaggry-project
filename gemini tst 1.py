
import streamlit as st
import psycopg2
import pandas as pd
import requests
import json
from datetime import datetime
import sys
import matplotlib.pyplot as plt # <-- NEW IMPORT
import seaborn as sns           # <-- NEW IMPORT
import numpy as np

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
- If the user asks for analysis, correlations, or plots for a range (e.g., "analyze batch 1 to 20" or "heatmap for batches 50 to 1000"), just generate the query to SELECT ALL data for that range (e.g., "SELECT * FROM batch_data WHERE batch_id BETWEEN 1 AND 20;"). The Python script will handle the analysis.

User Question: "{{question}}"
SQL Query:
"""

# ▼▼▼ MODIFIED RAG PROMPT ▼▼▼
# This prompt now instructs the LLM to simply "present" the analysis
# that Python has already completed, rather than re-interpreting the raw data.
RAG_PROMPT_TEMPLATE = """
You are a helpful assistant for the Jaggery Batch production team.
A Python script has already performed a detailed analysis on the requested data.
Your task is to present this analysis summary to the user in a clear and conversational way.

# Analysis Summary from Python:
{results}

# User Question:
"{question}"

Answer:
(Present the summary conversationally. Start by answering the question directly.
For example: "Yes, I've analyzed the data for batches... Here's what I found:"
Mention that the plots are displayed below the text.)
"""
# ▲▲▲ END MODIFIED RAG PROMPT ▲▲▲

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="🏭 Jaggery Batch Assistant", layout="centered")
st.title("🏭 Jaggery Batch Assistant")
st.caption(f"Conversational interface for TimescaleDB, powered by Ollama ({MODEL_NAME}).")

# -------------------------------
# Database Connection
# -------------------------------

@st.cache_resource
def init_connection():
    """Initialize connection to TimescaleDB."""
    try:
        conn = psycopg2.connect(
            dbname="postgres",
            user="postgres",
            password="aura",  # change to your password
            host="localhost",
            port=5439
        )
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
        df = convert_timestamps(df)
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
    
    if not sql_query.endswith(';'):
        sql_query += ';'

    if not sql_query.lstrip().upper().startswith("SELECT"):
        st.error("❌ Unsafe SQL generated (not SELECT). Try rephrasing.")
        return None

    return sql_query

# ▼▼▼ NEW SECTION: Matplotlib/Seaborn Analysis ▼▼▼
# -------------------------------
# Analysis & Plotting Function
# -------------------------------

def perform_analysis_and_plot(df_input):
    """
    Calculates metrics, generates Matplotlib/Seaborn plots,
    and returns a text summary.
    Plots are displayed directly using st.pyplot().
    """
    summary = ""
    df = df_input.copy()

    # 1. Calculate Delay
    if 'end_time' not in df.columns or 'next_batch_start' not in df.columns:
        return "Cannot perform analysis: 'end_time' or 'next_batch_start' columns are missing."

    if not pd.api.types.is_datetime64_any_dtype(df['end_time']):
         df['end_time'] = pd.to_datetime(df['end_time'], errors='coerce')
    if not pd.api.types.is_datetime64_any_dtype(df['next_batch_start']):
         df['next_batch_start'] = pd.to_datetime(df['next_batch_start'], errors='coerce')

    df = df.dropna(subset=['end_time', 'next_batch_start'])
    if df.empty:
        return "Not enough valid timestamp data to calculate delays."

    df['delay_seconds'] = (df['next_batch_start'] - df['end_time']).dt.total_seconds()
    
    # 2. Calculate Duration (if possible)
    if 'start_time' in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df['start_time']):
            df['start_time'] = pd.to_datetime(df['start_time'], errors='coerce')
        df = df.dropna(subset=['start_time'])
        if not df.empty:
            df['duration_seconds'] = (df['end_time'] - df['start_time']).dt.total_seconds()

    # 3. Classify Delay
    bins = [-float('inf'), 10, 60, 900, float('inf')]
    labels = ['Perfect (≤10s)', 'Normal (10-60s)', 'Cleaning Delay (60-900s)', 'Failure (>900s)']
    df['delay_category'] = pd.cut(df['delay_seconds'], bins=bins, labels=labels, right=True)
    
    category_counts = df['delay_category'].value_counts().sort_index()
    summary += "Delay Classification Counts:\n"
    summary += category_counts.to_string() + "\n\n"
    
    summary += "Delay Statistics (in seconds):\n"
    summary += df['delay_seconds'].describe().to_string() + "\n\n"

    # --- Generate Plots ---
    sns.set_theme(style="whitegrid")

    # Plot 1: Histogram of Delays (Good for 1000+ batches)
    st.write("### Delay Distribution (Histogram)")
    fig1, ax1 = plt.subplots()
    sns.histplot(df['delay_seconds'].clip(upper=1000), bins=50, kde=True, ax=ax1) # Clip at 1000s for readability
    ax1.set_title('Distribution of Batch Delays (Clipped at 1000s)')
    ax1.set_xlabel('Delay (seconds)')
    ax1.set_ylabel('Frequency (Number of Batches)')
    st.pyplot(fig1)

    # Plot 2: Bar chart of Delay Categories
    st.write("### Batch Count by Delay Category")
    fig2, ax2 = plt.subplots()
    category_counts.plot(kind='bar', ax=ax2, color=['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728'])
    ax2.set_title('Total Batches by Delay Category')
    ax2.set_xlabel('Delay Category')
    ax2.set_ylabel('Number of Batches')
    ax2.tick_params(axis='x', rotation=0)
    st.pyplot(fig2)

    # Plot 3: Correlation heatmap
    st.write("### Correlation Heatmap")
    corr_cols = [col for col in ['batch_id', 'duration_seconds', 'delay_seconds'] 
                 if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
    
    if len(corr_cols) > 1:
        corr_df = df[corr_cols].corr()
        
        fig3, ax3 = plt.subplots()
        sns.heatmap(corr_df, annot=True, fmt='.2f', cmap='vlag', center=0, ax=ax3)
        ax3.set_title('Correlation Heatmap')
        st.pyplot(fig3)
        
        summary += "Correlation Matrix:\n"
        summary += corr_df.to_string() + "\n"
    else:
        summary += "Could not calculate correlation (missing duration or delay data).\n"

    return summary

# ▲▲▲ END NEW SECTION ▲▲▲

# -------------------------------
# Chat Interface
# -------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "👋 Hi sir! I’m your Jaggery Batch Assistant. Ask me about batch times, delays, or status. \n\nTry 'Analyze batches 1 to 1000' to see the new features!"
    }]

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Plots are now generated live, so they won't be in history.

# ▼▼▼ MODIFIED CHAT LOOP ▼▼▼
if prompt := st.chat_input("Ask about batch data..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # 1. Generate SQL
            sql_query = generate_sql(prompt)
            if not sql_query:
                answer = "I couldn’t generate a valid SQL query. Please rephrase your question."
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.stop() 

            with st.expander("🔍 View Generated SQL Query"):
                st.code(sql_query, language="sql")

            # 2. Run Query
            df = run_query(sql_query)
            
            results_for_llm = ""
            
            if "error" in df.columns:
                error_message = df["error"].iloc[0]
                st.error(f"SQL Error: {error_message}")
                results_for_llm = {"error": error_message}
            elif df.empty:
                results_for_llm = "No data found for this query."
            else:
                # 3. Check for Analysis Intent
                analysis_keywords = ["analyze", "analysis", "correlation", "plot", "visualize", "relation", "chart", "heatmap", "delay analysis", "describe"]
                is_analysis_request = any(kw in prompt.lower() for kw in analysis_keywords)

                if is_analysis_request:
                    st.write(f"--- \n### 📈 Analysis for {df.shape[0]} Batches\n---")
                    # This function now PRINTS plots directly
                    # and returns the text summary for the LLM.
                    analysis_summary = perform_analysis_and_plot(df)
                    results_for_llm = analysis_summary
                else:
                    # Original behavior: just show the data
                    results_for_llm = df.to_markdown(index=False)

            # 4. Generate RAG Answer
            rag_prompt = RAG_PROMPT_TEMPLATE.format(results=results_for_llm, question=prompt)
            answer = ask_ollama(rag_prompt)

            if answer is None:
                answer = "I retrieved the data, but encountered an issue analyzing it with the LLM."

            # 5. Display Answer
            st.markdown(answer)
            # Plots are already displayed by perform_analysis_and_plot()

    # Add only the text answer to history
    st.session_state.messages.append({"role": "assistant", "content": answer})
# ▲▲▲ END MODIFIED CHAT LOOP ▲▲▲
