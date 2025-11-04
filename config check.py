# -------------------------------
# Full Test Script: PostgreSQL + Ollama + Streamlit
# -------------------------------

import streamlit as st
import psycopg2
import pandas as pd
import requests
import json
import sys

st.title("🧪 Jaggery System Test")
st.caption("Tests TimescaleDB connection and Ollama API functionality")

# -------------------------------
# 1️⃣ Show Python Executable
# -------------------------------
st.write("Python executable being used:")
st.code(sys.executable)

# -------------------------------
# 2️⃣ Test PostgreSQL Connection
# -------------------------------
def test_db_connection():
    try:
        conn = psycopg2.connect(
            dbname="postgres",
            user="postgres",
            password="aura",  # change if needed
            host="localhost",
            port=5439
        )
        st.success("✅ TimescaleDB/PostgreSQL connection successful")
        return conn
    except Exception as e:
        st.error(f"❌ Failed to connect to TimescaleDB/PostgreSQL:\n{e}")
        return None

conn = test_db_connection()

# -------------------------------
# 3️⃣ Test SQL Query
# -------------------------------
if conn:
    try:
        st.write("Running test query: SELECT batch_id, start_time FROM batch_data LIMIT 5;")
        df = pd.read_sql("SELECT batch_id, start_time FROM batch_data LIMIT 5;", conn)
        st.write("Query Result:")
        st.dataframe(df)
        if df.empty:
            st.warning("⚠️ The table returned 0 rows. Check if 'batch_data' table has data.")
    except Exception as e:
        st.error(f"❌ SQL Query failed: {e}")

# -------------------------------
# 4️⃣ Test Ollama API
# -------------------------------
OLLAMA_HOST = "http://localhost:11434"
MODEL_NAME = "granite3.3:8b"

def test_ollama():
    prompt = "Write a safe SQL SELECT query to get the first 5 rows from batch_data."
    st.write("Sending prompt to Ollama API:")
    st.code(prompt)
    try:
        response = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={"model": MODEL_NAME, "prompt": prompt, "stream": False},
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        result_text = data.get("response", "").strip()
        st.success("✅ Ollama API call successful")
        st.write("Response from Ollama:")
        st.code(result_text)
    except requests.exceptions.ConnectionError:
        st.error(f"❌ Could not connect to Ollama at {OLLAMA_HOST}. Make sure `ollama serve` is running.")
    except Exception as e:
        st.error(f"❌ Ollama API call failed: {e}")

test_ollama()
