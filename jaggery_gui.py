# -------------------------------
# Ollama + TimescaleDB RAG Script for Streamlit
# -------------------------------

import streamlit as st
import psycopg2
import pandas as pd
import subprocess

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
st.caption("Your conversational interface for TimescaleDB batch data, powered by Ollama.")


# -------------------------------
# 1️⃣ Connect to TimescaleDB
# -------------------------------
# Use st.cache_resource to only run this connection once.
@st.cache_resource
def init_connection():
    """Initialize the connection to TimescaleDB."""
    try:
        # NOTE: The connection details are hardcoded here for simplicity.
        # For production environments, it's recommended to use Streamlit secrets.
        # Create a file .streamlit/secrets.toml and add your credentials there.
        conn = psycopg2.connect(
            dbname="postgres",
            user="postgres",
            password="****", # change to actual password !!
            host="localhost",
            port=5439
        )
        return conn
    except Exception as e:
        st.error(f"Failed to connect to the database. Please ensure your TimescaleDB is running and the connection details are correct.\n\n**Error:** {e}")
        st.stop()

conn = init_connection()

def run_query(sql_query):
    """Execute SQL query safely and return results as DataFrame."""
    try:
        df = pd.read_sql(sql_query, conn)
        return df
    except Exception as e:
        # Return a DataFrame with an error message to handle it gracefully downstream
        return pd.DataFrame([{'error': str(e)}])

# -------------------------------
# 2️⃣ Ask Ollama via CLI
# -------------------------------
# Use st.cache_data for functions that return serializable data.
@st.cache_data(show_spinner=False)
def ask_ollama(prompt, model_name="granite3.3:8b"):
    """Ask Ollama model via CLI and get response (UTF-8 safe)."""
    try:
        result = subprocess.run(
            ["ollama", "run", model_name],
            input=prompt,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace"
        )
        if result.returncode != 0:
            return f"Ollama CLI Error: {result.stderr}"
        return result.stdout.strip()
    except FileNotFoundError:
        return "Error: The 'ollama' command was not found. Please ensure Ollama is installed and in your system's PATH."
    except Exception as e:
        return f"An unexpected error occurred: {e}"

# -------------------------------
# 3️⃣ Generate SQL from Natural Language
# -------------------------------
def generate_sql(question):
    """Use Ollama to convert natural language into a SQL query."""
    # This prompt is carefully designed for SQL generation.
    prompt = f"""
You are an expert data assistant with access to a PostgreSQL table named 'batch_data'.

we have only fue data so answer only from the database , we dont need any extra information.

only give answer no advice ,,,

This table has the following columns: batch_id, start_time (timestamp), end_time (timestamp),
next_batch_start (timestamp), gap_delay (text), and status (text).

The 'gap_delay' column can only contain one of four string values: 'normal', 'failure', 'perfect', 'cleaning_delay'.

Your task is to convert the user's question into a safe, and syntactically correct SQL SELECT query for PostgreSQL.
- Only generate the SQL query. Do not add any explanations, introductions, or markdown formatting like '```sql'.
- Use proper timestamp operations like BETWEEN, >=, <=. Do not use string concatenation or LIKE on timestamps.
- Ensure all string comparisons for 'gap_delay' or 'status' use single quotes.



User Question: "{question}"
SQL Query:
"""
    sql_query = ask_ollama(prompt)
    # Clean up the response to ensure it's just the SQL
    sql_query = sql_query.strip().strip('`').strip()
    if sql_query.lower().startswith('sql'):
        sql_query = sql_query[3:].strip()
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
    
    # Display the generated SQL in an expander for debugging/transparency
    with st.expander("🔍 View Generated SQL Query"):
        st.code(sql_query, language="sql")

    # Step 2: Run the query against the database
    df = run_query(sql_query)
    
    # Handle SQL errors gracefully
    if 'error' in df.columns:
        error_message = df['error'].iloc[0]
        st.error(f"There was an error with the generated SQL. Please try rephrasing your question.\n\n**Error details:** {error_message}")
        return "I couldn't process your request due to a database error."

    # Step 3: Use the query results to generate a natural language answer
    results = df.to_dict(orient='records')
    rag_prompt = f"""
You are a friendly and helpful assistant for the Jaggery Batch production team.
Based on the following data retrieved from the database, please answer the user's question.
If the data is empty, inform the user that no records were found that match their criteria.

Database Results:
{results}

User's Question:
"{question}"

Answer:
"""
    return ask_ollama(rag_prompt)

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

