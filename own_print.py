#print all the necessary information

import json
import faiss
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
import string
from rank_bm25 import BM25Okapi
import spacy

# Load Swedish spaCy model for lemmatization
nlp = spacy.load("sv_core_news_sm")

# Load data and preprocess
with open('output.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize
    tokens = word_tokenize(text)
    # Lemmatization using spaCy
    lemmatized_tokens = [token.lemma_ for token in nlp(" ".join(tokens))]
    print("Lemmatized Tokens:", lemmatized_tokens)  # Print lemmatized tokens
    # Remove stopwords
    stop_words = set(stopwords.words('swedish'))
    tokens = [word for word in lemmatized_tokens if word not in stop_words]
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
index = faiss.IndexFlatL2(data_embeddings.shape[1])
index.add(data_embeddings)

# Create BM25 index
tokenized_data = [word_tokenize(doc) for doc in data_processed]
bm25_index = BM25Okapi(tokenized_data)

# User interaction
while True:
    query = input("Enter your query (or 'q' to quit): ")
    if query == 'q':
        break
    
    # Expand query with synonyms or related concepts using word embeddings (modify this part according to the chosen method)
    expanded_query = query + " synonyms"  # Example query expansion
    print("Expanded Query:", expanded_query)
    
    query_processed = preprocess_query(expanded_query)
    print("Processed Query:", query_processed)
    
    query_embedding = model.encode([query_processed])
    print("Query Embedding:", query_embedding)

    # Search with Faiss
    D, I = index.search(query_embedding, k=3)  # Adjust k as needed
    print("Faiss Search Results (Distances):", D)
    print("Faiss Search Results (Indices):", I)

    # Refine results with BM25
    bm25_scores = bm25_index.get_scores(query_processed)
    print("BM25 Scores:", bm25_scores)

    combined_scores = [(i, d + bm25_scores[i]) for i, d in zip(I[0], D[0])]
    combined_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Display results
    print("Search Results:")
    for i, score in combined_scores:
        print(f"Combined Score: {score:.4f}")
        print(f"Title: {data[i]['title']}")
        print(f"URL: {data[i]['url']}")
        print(f"Summary: {data[i]['summery']}")
        print(f"Content: {data[i]['content']}")
        print("---")
