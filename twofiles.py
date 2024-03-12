import json
import faiss
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
import string
from rank_bm25 import BM25Okapi
import time
import pickle
import torch

# Load data and preprocess
with open('migration.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    

def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('swedish'))
    tokens = [word for word in tokens if word not in stop_words]
    # Join tokens back into string
    processed_text = ' '.join(tokens)
    return processed_text

# Preprocess function for queries
def preprocess_query(query):
    return preprocess_text(query)

# Process data
data_processed = [preprocess_text(entry['content']) for entry in data]
time1 = time.time()
# Load pre-trained model
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

# Generate embeddings
data_embeddings = model.encode(data_processed)
print(data_embeddings)
time2 = time.time()
print('Time Taken for embedding: ', time2-time1)
index = faiss.IndexFlatL2(data_embeddings.shape[1])
index.add(data_embeddings)

# Save Faiss index
faiss.write_index(index, 'faiss_index.bin')

# Create BM25 index
tokenized_data = [word_tokenize(doc) for doc in data_processed]
bm25_index = BM25Okapi(tokenized_data)

# Save BM25 index
with open('bm25_index.pkl', 'wb') as file:
    pickle.dump(bm25_index, file)

print('Time for indexing:', time.time()-time2)

# User interaction
while True:
    query = input("Enter your query (or 'q' to quit): ")
    if query == 'q':
        break
    query_processed = preprocess_query(query)
    query_embedding = model.encode([query_processed])

    # Load Faiss index
    index = faiss.read_index('faiss_index.bin')

    # Search with Faiss
    D, I = index.search(query_embedding, k=3)  # Adjust k as needed

    # Load BM25 index
    with open('bm25_index.pkl', 'rb') as file:
        bm25_index = pickle.load(file)
    bm25_scores = bm25_index.get_scores(query_processed)
    combined_scores = [(i, d + bm25_scores[i]) for i, d in zip(I[0], D[0])]
    combined_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Display results
    print("Search Results:")
    for i, score in combined_scores:
        print(f"Combined Score: {score:.4f}")
        print(f"Title: {data[i]['title']}")
        print(f"URL: {data[i]['url']}")
        print(f"Summary: {data[i]['summary']}")
        print(f"Content: {data[i]['content']}")
        print("---")
