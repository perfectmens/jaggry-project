# -------------------------------
# Ollama + TimescaleDB RAG Script (Fixed for Windows)
# -------------------------------

import psycopg2
import pandas as pd
import subprocess

# -------------------------------
# 1️⃣ Connect to TimescaleDB
# -------------------------------
DB_CONFIG = {
    'dbname': 'postgres',        # since you don’t know dbname, postgres is default
    'user': 'postgres',
    'password': 'aura',
    'host': 'localhost',
    'port': 5439                # 5432 is default for PostgreSQL
}

conn = psycopg2.connect(**DB_CONFIG)

def run_query(sql_query):
    """Execute SQL query safely and return results as DataFrame."""
    try:
        df = pd.read_sql(sql_query, conn)
        return df
    except Exception as e:
        return pd.DataFrame([{'error': str(e)}])

# -------------------------------
# 2️⃣ Ask Ollama via CLI
# -------------------------------
def ask_ollama(prompt, model_name="granite3.3:8b"):
    """Ask Ollama model via CLI and get response (UTF-8 safe)."""
    try:
        result = subprocess.run(
            ["ollama", "run", model_name],
            input=prompt,                # pass prompt directly as str
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace"
        )
        if result.returncode != 0:
            return f"Ollama CLI Error: {result.stderr}"
        return result.stdout.strip()
    except Exception as e:
        return f"Error: {e}"

# -------------------------------
# 3️⃣ Generate SQL from NL
# -------------------------------
def generate_sql(question):
    """Use Ollama to convert natural language → SQL."""
    prompt = f"""
You are a data assistant with access to a PostgreSQL table 'batch_data' 
with the following columns: batch_id, start_time (timestamp), end_time (timestamp), 
next_batch_start, gap_delay, status.

gap_delay row contain only four things : 'normal', 'failure', 'perfect', 'cleaning_delay'.

Convert the user's question into a safe SQL SELECT query.
Use proper timestamp operations: BETWEEN, >=, <=. Do not use string concatenation or LIKE on timestamps.
Return SQL only.

User Question: {question}
"""

    sql_query = ask_ollama(prompt)
    sql_query = sql_query.strip().strip('```').strip()
    return sql_query

# -------------------------------
# 4️⃣ Full Flow: Ask Question
# -------------------------------
def answer_question(question):
    sql_query = generate_sql(question)
    print(f"\n[DEBUG] Generated SQL:\n{sql_query}\n")

    df = run_query(sql_query)
    if 'error' in df.columns:
        return f"SQL Error: {df['error'].iloc[0]}"

    results = df.to_dict(orient='records')
    rag_prompt = f"""
SQL query results:
{results}

Answer the user's question like a friendly assistant:
{question}
"""
    return ask_ollama(rag_prompt)

# -------------------------------
# 5️⃣ Main Loop
# -------------------------------
if __name__ == "__main__":
    print("Welcome to Jaggery Batch Assistant (Ollama + TimescaleDB)\n")
    while True:
        user_question = input("Ask a question (or type 'exit'): ")
        if user_question.lower() in ['exit', 'quit']:
            break
        answer = answer_question(user_question)
        print(f"\nAnswer:\n{answer}\n{'-'*60}\n")
