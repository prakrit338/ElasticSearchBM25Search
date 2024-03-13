from flask import Flask, render_template, request
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

app = Flask(__name__)

# Load data and preprocess
with open('my_output.json', 'r', encoding='utf-8') as f:
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

# Load pre-trained model
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

# Generate embeddings
data_embeddings = model.encode(data_processed)
faiss_index = faiss.IndexFlatL2(data_embeddings.shape[1])
faiss_index.add(data_embeddings)

# Create BM25 index
tokenized_data = [word_tokenize(doc) for doc in data_processed]
bm25_index = BM25Okapi(tokenized_data)

# Create BM25 index
tokenized_data = [word_tokenize(doc) for doc in data_processed]
bm25_index = BM25Okapi(tokenized_data)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    query_processed = preprocess_query(query)
    query_embedding = model.encode([query_processed])

    # Search with Faiss
    D, I = faiss_index.search(query_embedding, k=3)  # Adjust k as needed

    # Load BM25 index
    with open('bm25_index.pkl', 'rb') as file:
        bm25_index = pickle.load(file)
    bm25_scores = bm25_index.get_scores(query_processed)
    combined_scores = [(i, d + bm25_scores[i]) for i, d in zip(I[0], D[0])]
    combined_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Prepare search results
    search_results = []
    for i, score in combined_scores:
        result = {
            'title': data[i]['title'],
            'url': data[i]['url'],
            # 'summary': data[i]['summary'],
            'content': data[i]['content']
        }
        search_results.append(result)

    return render_template('results.html', query=query, results=search_results)

if __name__ == '__main__':
    app.run(debug=True)
