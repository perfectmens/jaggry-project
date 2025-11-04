# -------------------------------
# Ollama + TimescaleDB RAG Script for Streamlit (Refactored)
# -------------------------------

import streamlit as st
import psycopg2
import pandas as pd
import requests  # Use requests for API calls, not subprocess
import json

# -------------------------------
# Constants
# -------------------------------

# Define your model and Ollama host
MODEL_NAME = "granite3.3:8b"
OLLAMA_HOST = "http://localhost:11434"

# Define the DB schema and prompts as constants for easy maintenance
DB_SCHEMA_INFO = """
This table has the following columns: batch_id, start_time (timestamp), end_time (timestamp),
next_batch_start (timestamp), gap_delay (text), and status (text).

The 'gap_delay' column can only contain one of four string values: 'normal', 'failure', 'perfect', 'cleaning_delay'.
"""

SQL_GEN_PROMPT_TEMPLATE = f"""
You are an expert data assistant with access to a PostgreSQL table named 'batch_data'.
{DB_SCHEMA_INFO}

Your task is to convert the user's question into a safe, and syntactically correct SQL SELECT query for PostgreSQL.
- Only generate the SQL query. Do not add any explanations, introductions, or markdown formatting like '```sql'.
- Use proper timestamp operations like BETWEEN, >=, <=. Do not use string concatenation or LIKE on timestamps.
- Ensure all string comparisons for 'gap_delay' or 'status' use single quotes.
- ONLY output SELECT queries. Never output any other command (UPDATE, DELETE, DROP, etc.).

User Question: "{{question}}"
SQL Query:
"""

RAG_PROMPT_TEMPLATE = """
You are a friendly and helpful assistant for the Jaggery Batch production team.
Based on the following data retrieved from the database, please answer the user's question.
If the data is empty, inform the user that no records were found that match their criteria.
If the data contains an error, inform the user about the error.

Database Results:
{{results}}

User's Question:
"{{question}}"

Answer:
"""

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Jaggery Batch Assistant",
    page_icon="🏭",
    layout="centered",
    initial_sidebar_state="auto"
)

st.title("🏭 Jaggery Batch Assistant")
st.caption(f"Conversational interface for TimescaleDB, powered by Ollama ({MODEL_NAME}).")


# -------------------------------
# 1️⃣ Connect to TimescaleDB (Using Streamlit Secrets)
# -------------------------------

# Create a file .streamlit/secrets.toml with your credentials:
#
# [db_credentials]
# host = "localhost"
# port = 5439
# dbname = "postgres"
# user = "postgres"
# password = "YOUR_PASSWORD_HERE"
#
def init_connection():
    """Initialize the connection to TimescaleDB without caching for testing."""
    try:
        st.write("Trying to connect to PostgreSQL...")
        conn = psycopg2.connect(
            dbname="postgres",
            user="postgres",
            password="aura",  # your password
            host="localhost",
            port=5439
        )
        st.success("Database connected ✅")
        return conn
    except Exception as e:
        st.error(f"DB Connection Error: {e}")
        st.stop()

conn = init_connection()
# -------------------------------

def run_query(sql_query):
    """Execute SQL query safely and return results as DataFrame."""
    try:
        df = pd.read_sql(sql_query, conn)
        return df
    except Exception as e:
        # Return a DataFrame with an error message to handle it gracefully downstream
        return pd.DataFrame([{'error': str(e)}])

# -------------------------------
# 2️⃣ Ask Ollama via REST API
# -------------------------------
@st.cache_data(show_spinner=False)
def ask_ollama(full_prompt, model_name=MODEL_NAME):
    """Ask Ollama model via REST API and get a non-streaming response."""
    url = f"{OLLAMA_HOST}/api/generate"
    payload = {
        "model": model_name,
        "prompt": full_prompt,
        "stream": False  # We want the full response at once
    }
    try:
        # Set a reasonable timeout (e.g., 60 seconds)
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        data = response.json()
        return data.get("response", "").strip()
        
    except requests.exceptions.ConnectionError:
        st.error(f"Ollama Connection Error: Could not connect to {OLLAMA_HOST}. Please ensure Ollama is running (`ollama serve`).")
        return None  # Return None to indicate failure
    except requests.exceptions.RequestException as e:
        st.error(f"Ollama API Error: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred while querying Ollama: {e}")
        return None

# -------------------------------
# 3️⃣ Generate SQL from Natural Language
# -------------------------------
def generate_sql(question):
    """Use Ollama to convert natural language into a SQL query."""
    prompt = SQL_GEN_PROMPT_TEMPLATE.format(question=question)
    sql_query = ask_ollama(prompt)

    if sql_query is None:  # Handle API error from ask_ollama
        return None

    # --- SQL Safety Cleanup & Validation ---
    sql_query = sql_query.strip().strip('`').strip()
    if sql_query.lower().startswith('sql'):
        sql_query = sql_query[3:].strip()
    
    # CRITICAL: Basic SQL safety guardrail
    if not sql_query.lstrip().upper().startswith("SELECT"):
        st.error(f"Generated query was unsafe (not a SELECT query). Please rephrase your question.")
        return None

    return sql_query

# -------------------------------
# 4️⃣ Full RAG Flow: Ask Question and Get Answer
# -------------------------------
def answer_question(question):
    """
    Takes a user's question, generates SQL, queries the DB,
    and then uses the results to generate a final answer.
    """
    # Step 1: Generate SQL from the question
    sql_query = generate_sql(question)
    
    if not sql_query:  # Handles None from API error or safety check
        return "I was unable to generate a safe SQL query. Please try rephrasing your question."

    # Display the generated SQL in an expander for debugging/transparency
    with st.expander("🔍 View Generated SQL Query"):
        st.code(sql_query, language="sql")

    # Step 2: Run the query against the database
    df = run_query(sql_query)
    
    # Handle SQL errors gracefully (run_query returns a df with 'error' column)
    if 'error' in df.columns:
        error_message = df['error'].iloc[0]
        st.error(f"There was an error with the generated SQL. Please try rephrasing your question.\n\n**Error details:** {error_message}")
        # Feed the error back to the LLM so it can explain it
        results = {"error": error_message}
    else:
        results = df.to_dict(orient='records')

    # Step 3: Use the query results to generate a natural language answer
    rag_prompt = RAG_PROMPT_TEMPLATE.format(results=results, question=question)
    
    response = ask_ollama(rag_prompt)
    
    if response is None:
        return "I successfully retrieved the data, but had an issue analyzing it with the LLM. Please try again."
    
    return response

# -------------------------------
# 5️⃣ Streamlit Chat UI
# -------------------------------

# Initialize chat history in session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Hi! I'm your Jaggery Batch Assistant. How can I help you analyze the batch data today?"
    }]

# Display past messages from the chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Main chat input loop
if prompt := st.chat_input("Ask about batch times, delays, or status..."):
    # Add user's message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user's message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display assistant's response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = answer_question(prompt)
            st.markdown(response)
    
    # Add assistant's response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

