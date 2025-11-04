from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama
from mock_data import get_mock_batch_data

# --- Application Setup ---
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Data Models ---
class Query(BaseModel):
    question: str

# --- Global Variables & Model Loading ---
# We load the models and data once at startup to avoid reloading on every request
try:
    with open('backend/embeddings.pkl', 'rb') as f:
        data = pickle.load(f)
        sentences = data['sentences']
        embeddings = data['embeddings']

    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Embeddings and model loaded successfully.")

except FileNotFoundError:
    print("ERROR: embeddings.pkl not found. Please run process_data.py first.")
    sentences, embeddings, model = [], None, None

# --- RAG Core Logic ---
def find_relevant_context(query_embedding, top_k=3):
    """
    Finds the most relevant sentences from the knowledge base using cosine similarity.
    """
    if embeddings is None or len(embeddings) == 0:
        return []

    # Calculate cosine similarity
    cos_scores = np.dot(embeddings, query_embedding) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding))

    # Get the top_k most similar sentences
    top_k_indices = np.argsort(cos_scores)[-top_k:][::-1]

    return [sentences[i] for i in top_k_indices]

# --- API Endpoints ---
@app.get("/")
def read_root():
    return {"message": "Welcome to the RAG API!"}

@app.post("/api/query")
def query_rag(query: Query):
    """
    The main endpoint for the RAG system.
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Please check server logs.")

    # 1. Embed the user's question
    query_embedding = model.encode(query.question)

    # 2. Find the most relevant context
    context = find_relevant_context(query_embedding)

    if not context:
        return {"answer": "I'm sorry, I couldn't find any relevant information to answer your question.", "sources": []}

    # 3. Format the prompt for the LLM
    prompt = f"Using the following context, please answer the question.\n\nContext:\n{' '.join(context)}\n\nQuestion: {query.question}"

    try:
        # 4. Send to Ollama
        response = ollama.chat(
            model='granite-8b-base', # Using the model you specified
            messages=[{'role': 'user', 'content': prompt}]
        )
        answer = response['message']['content']
        return {"answer": answer, "sources": context}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to communicate with Ollama: {str(e)}")

@app.get("/api/batch-data")
def get_batch_data():
    """
    Provides the raw mock data for frontend visualizations.
    """
    return get_mock_batch_data()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
