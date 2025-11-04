import pickle
from sentence_transformers import SentenceTransformer
from mock_data import get_mock_batch_data

def main():
    """
    Processes the mock data, generates embeddings, and saves them to a file.
    """
    # 1. Get the mock data
    batch_data = get_mock_batch_data()
    print(f"Loaded {len(batch_data)} mock data records.")

    # 2. Transform data into sentences
    sentences = [
        f"Batch {row[0]} started at {row[1]}, ended at {row[2]}, and the next batch started at {row[3]}."
        for row in batch_data
    ]

    # 3. Generate embeddings
    print("Generating embeddings for mock data...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(sentences)
    print("Embeddings generated successfully.")

    # 4. Save the sentences and embeddings to a file
    #    We'll use this file in our RAG system as a simple vector store.
    with open('backend/embeddings.pkl', 'wb') as f:
        pickle.dump({'sentences': sentences, 'embeddings': embeddings}, f)

    print("Sentences and embeddings have been saved to backend/embeddings.pkl")

if __name__ == "__main__":
    main()
