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
import altair as alt  # <-- NEW IMPORT for plotting
import numpy as np   # <-- NEW IMPORT for correlation

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
- If the user asks for analysis, correlations, or plots for a range (e.g., "analyze batch 1 to 20"), just generate the query to SELECT ALL data for that range (e.g., "SELECT * FROM batch_data WHERE batch_id BETWEEN 1 AND 20;"). The Python script will handle the analysis.

User Question: "{{question}}"
SQL Query:
"""

# ▼▼▼ NEW/MODIFIED SECTION ▼▼▼
RAG_PROMPT_TEMPLATE = """
You are a helpful assistant for the Jaggery Batch production team.
Based on the following database data and analysis summary, answer the user's question clearly.
If there is no data, say so. If there's an error, describe it briefly.

# Data Context:
- We have performed an analysis on the data.
- 'delay_seconds' is the calculated gap in seconds (next_batch_start - end_time).
- Delay Categories are defined as:
  - Perfect: delay <= 10s
  - Normal: 10s < delay <= 60s
  - Cleaning Delay: 60s < delay <= 900s
  - Failure: delay > 900s

# Database Results & Analysis Summary:
{results}

# User Question:
"{question}"

Answer:
(Provide a clear, concise answer based on the summary. Mention that the plots are displayed below.)
"""
# ▲▲▲ END NEW/MODIFIED SECTION ▲▲▲

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
        # st.write("🔌 Connecting to PostgreSQL...") # Let's hide this for a cleaner chat
        conn = psycopg2.connect(
            dbname="postgres",
            user="postgres",
            password="aura",  # change to your password
            host="localhost",
            port=5439
        )
        # st.success("✅ Database connected successfully") # And this
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
    
    # Ensure it ends with a semicolon
    if not sql_query.endswith(';'):
        sql_query += ';'

    if not sql_query.lstrip().upper().startswith("SELECT"):
        st.error("❌ Unsafe SQL generated (not SELECT). Try rephrasing.")
        return None

    return sql_query

# ▼▼▼ NEW SECTION ▼▼▼
# -------------------------------
# Analysis & Plotting Function
# -------------------------------

def perform_analysis(df_input):
    """
    Calculates metrics, generates Altair plots, and returns a text summary.
    """
    plots = {}
    summary = ""
    df = df_input.copy()  # Work on a copy

    # 1. Calculate Delay
    if 'end_time' not in df.columns or 'next_batch_start' not in df.columns:
        return "Cannot perform analysis: 'end_time' or 'next_batch_start' columns are missing.", {}

    # Ensure columns are datetime (they should be, but double-check)
    if not pd.api.types.is_datetime64_any_dtype(df['end_time']):
         df['end_time'] = pd.to_datetime(df['end_time'], errors='coerce')
    if not pd.api.types.is_datetime64_any_dtype(df['next_batch_start']):
         df['next_batch_start'] = pd.to_datetime(df['next_batch_start'], errors='coerce')

    df = df.dropna(subset=['end_time', 'next_batch_start'])
    if df.empty:
        return "Not enough valid timestamp data to calculate delays.", {}

    df['delay_seconds'] = (df['next_batch_start'] - df['end_time']).dt.total_seconds()

    # 2. Classify Delay
    bins = [-float('inf'), 10, 60, 900, float('inf')]
    labels = ['Perfect (≤10s)', 'Normal (10-60s)', 'Cleaning Delay (60-900s)', 'Failure (>900s)']
    df['delay_category'] = pd.cut(df['delay_seconds'], bins=bins, labels=labels, right=True)
    
    summary += "Delay Classification Counts:\n"
    summary += df['delay_category'].value_counts().sort_index().to_string() + "\n\n"
    
    # --- Generate Plots ---
    delay_domain = ['Perfect (≤10s)', 'Normal (10-60s)', 'Cleaning Delay (60-900s)', 'Failure (>900s)']
    delay_range = ['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728'] # Green, Blue, Orange, Red

    # Plot 1: Delay bar chart (delay_seconds per batch_id)
    if not df.empty:
        chart_bar = alt.Chart(df).mark_bar().encode(
            x=alt.X('batch_id:O', title='Batch ID', axis=None), # Hide axis labels, use tooltip
            y=alt.Y('delay_seconds:Q', title='Delay (seconds)'),
            color=alt.Color('delay_category:N', title="Delay Category",
                          scale=alt.Scale(domain=delay_domain, range=delay_range)),
            tooltip=['batch_id', 'delay_seconds', 'delay_category', 'start_time', 'end_time']
        ).properties(
            title="Delay (Seconds) per Batch"
        ).interactive()
        plots['delay_bar_chart'] = chart_bar

    # Plot 2: Average delay per category
    avg_delay_data = df.groupby('delay_category', observed=True)['delay_seconds'].mean().reset_index()
    if not avg_delay_data.empty:
        chart_avg_bar = alt.Chart(avg_delay_data).mark_bar().encode(
            x=alt.X('delay_category:N', title='Category', sort=labels),
            y=alt.Y('delay_seconds:Q', title='Average Delay (seconds)'),
            color=alt.Color('delay_category:N', legend=None,
                          scale=alt.Scale(domain=delay_domain, range=delay_range)),
            tooltip=['delay_category', 'delay_seconds']
        ).properties(
            title="Average Delay by Category"
        )
        plots['average_delay_chart'] = chart_avg_bar

    # Plot 3: Correlation heatmap
    if 'start_time' in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df['start_time']):
            df['start_time'] = pd.to_datetime(df['start_time'], errors='coerce')
        
        df = df.dropna(subset=['start_time'])
        if not df.empty:
            df['duration_seconds'] = (df['end_time'] - df['start_time']).dt.total_seconds()
            
            # Select only numeric columns for correlation
            corr_cols = [col for col in ['batch_id', 'duration_seconds', 'delay_seconds'] 
                         if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
            
            if len(corr_cols) > 1:
                corr_df = df[corr_cols].corr()
                corr_df_long = corr_df.reset_index().melt('index')
                
                heatmap = alt.Chart(corr_df_long).mark_rect().encode(
                    x=alt.X('index:N', title=None),
                    y=alt.Y('variable:N', title=None),
                    color=alt.Color('value:Q', title='Correlation', scale=alt.Scale(domain=[-1, 1], range='diverging')),
                    tooltip=['index', 'variable', 'value']
                ).properties(
                    title="Correlation Heatmap (Duration vs. Delay)"
                )
                
                text = heatmap.mark_text(baseline='middle').encode(
                    text=alt.Text('value:Q', format='.2f'),
                    color=alt.condition(
                        alt.datum.value > 0.5, 
                        alt.value('white'), 
                        alt.value('black')
                    )
                )
                plots['correlation_heatmap'] = heatmap + text
                
                summary += "Correlation Matrix:\n"
                summary += corr_df.to_string() + "\n"
            else:
                summary += "Could not calculate correlation (missing duration or delay data).\n"
    else:
        summary += "Could not calculate correlation (missing 'start_time' column).\n"

    return summary, plots

# ▲▲▲ END NEW SECTION ▲▲▲

# -------------------------------
# RAG Pipeline (Original, will be modified in the chat loop)
# -------------------------------
# Note: The 'answer_question' function is being merged into the main chat loop
# to handle the plotting logic more cleanly.

# -------------------------------
# Chat Interface
# -------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "👋 Hi sir! I’m your Jaggery Batch Assistant. Ask me about batch times, delays, or status. \n\nTry 'Analyze batches 1 to 20' to see the new features!"
    }]

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Note: Plots are not saved in history, only the text.
        # This is a limitation of storing complex objects in st.session_state.

# ▼▼▼ NEW/MODIFIED SECTION ▼▼▼
# User input
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
                st.stop() # Stop execution for this run

            with st.expander("🔍 View Generated SQL Query"):
                st.code(sql_query, language="sql")

            # 2. Run Query
            df = run_query(sql_query)
            
            results_for_llm = ""
            plots_to_display = {}
            
            if "error" in df.columns:
                error_message = df["error"].iloc[0]
                st.error(f"SQL Error: {error_message}")
                results_for_llm = {"error": error_message}
            elif df.empty:
                results_for_llm = "No data found for this query."
            else:
                # 3. Check for Analysis Intent
                analysis_keywords = ["analyze", "analysis", "correlation", "plot", "visualize", "relation", "chart", "heatmap", "delay analysis"]
                is_analysis_request = any(kw in prompt.lower() for kw in analysis_keywords)

                if is_analysis_request:
                    st.write("Performing analysis and generating charts...")
                    analysis_summary, plots_to_display = perform_analysis(df)
                    results_for_llm = f"Analysis Summary:\n{analysis_summary}\n\nRaw Data Sample (first 5 rows):\n{df.head().to_markdown(index=False)}"
                else:
                    # Original behavior: just show the data
                    results_for_llm = df.to_markdown(index=False)

            # 4. Generate RAG Answer
            rag_prompt = RAG_PROMPT_TEMPLATE.format(results=results_for_llm, question=prompt)
            answer = ask_ollama(rag_prompt)

            if answer is None:
                answer = "I retrieved the data, but encountered an issue analyzing it with the LLM."

            # 5. Display Answer and Plots
            st.markdown(answer)
            if plots_to_display:
                for title, chart in plots_to_display.items():
                    st.altair_chart(chart, use_container_width=True)

    # Add only the text answer to history
    st.session_state.messages.append({"role": "assistant", "content": answer})
# ▲▲▲ END NEW/MODIFIED SECTION ▲▲▲