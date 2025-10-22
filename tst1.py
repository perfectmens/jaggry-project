# -------------------------------
# Ollama + TimescaleDB RAG Script (Friendly Version)
# -------------------------------

import psycopg2
import pandas as pd
from ollama import OllamaClient


# -------------------------------
# 1️⃣ Connect to TimescaleDB
# -------------------------------
DB_CONFIG = {
    'dbname': 'batch_data',    # your database name
    'user': 'postgres',        # your database user
    'password': 'aura',        # your password
    'host': 'localhost',
    'port': 5439              # default PostgreSQL port
}

try:
    conn = psycopg2.connect(**DB_CONFIG)
    print("✅ Connected to database successfully!\n")
except Exception as e:
    print(f"❌ Failed to connect to database: {e}")
    exit(1)

def run_query(sql_query):
    """Execute SQL query safely and return results as DataFrame."""
    try:
        df = pd.read_sql(sql_query, conn)
        return df
    except Exception as e:
        return pd.DataFrame([{'error': str(e)}])

# -------------------------------
# 2️⃣ Initialize Ollama
# -------------------------------
ollama = Ollama(model="your_model_name")  # Replace with your Ollama model

# -------------------------------
# 3️⃣ Generate SQL from Natural Language
# -------------------------------
def generate_sql(question):
    """
    Convert user's question into a safe SQL SELECT query on 'batch_data'.
    """
    prompt = f"""
You are a friendly data assistant. You have access to a PostgreSQL table 'batch_data' 
with columns: batch_id, start_time, end_time, next_batch_start, gap_delay, status.

Convert this user's question into a valid SQL SELECT query. 
Return only SQL, no explanations.

Question: {question}
"""
    response = ollama.chat(prompt)
    sql_query = response.text.strip().strip('```')
    return sql_query

# -------------------------------
# 4️⃣ Answer question with RAG
# -------------------------------
def answer_question(question):
    try:
        # Generate SQL
        sql_query = generate_sql(question)
        print(f"[DEBUG] Generated SQL:\n{sql_query}\n")

        # Run query
        df = run_query(sql_query)

        if 'error' in df.columns:
            return f"Oh no, there was a database error: {df['error'].iloc[0]}"

        # Prepare prompt for Ollama to give friendly answer
        results_dict = df.to_dict(orient='records')
        rag_prompt = f"""
Hi! Here's the data from the database based on your question:

{results_dict}

Now answer the question clearly and in a friendly tone:
{question}
"""
        final_response = ollama.chat(rag_prompt)
        return final_response.text.strip()
    
    except Exception as e:
        return f"Oops, something went wrong: {e}"

# -------------------------------
# 5️⃣ User interaction loop
# -------------------------------
if __name__ == "__main__":
    print("👋 Welcome to your Jaggery Batch Assistant (Ollama + TimescaleDB)\n")
    try:
        while True:
            user_question = input("Ask me anything about the batches (or type 'exit'): ")
            if user_question.lower() in ['exit', 'quit']:
                print("Goodbye! 🍯")
                break
            answer = answer_question(user_question)
            print(f"\nAnswer:\n{answer}\n{'-'*50}\n")
    except KeyboardInterrupt:
        print("\nInterrupted! Exiting gracefully...")
