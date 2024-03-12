import json
import faiss
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
import string

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
model = SentenceTransformer('all-mpnet-base-v2')

# Generate embeddings
data_embeddings = model.encode(data_processed)
index = faiss.IndexFlatL2(data_embeddings.shape[1])
index.add(data_embeddings)

# User interaction
while True:
    query = input("Enter your query (or 'q' to quit): ")
    if query == 'q':
        break
    query_processed = preprocess_query(query)
    query_embedding = model.encode([query_processed])

    # Search with Faiss
    D, I = index.search(query_embedding, k=1)  # k nearest neighbors

    # Display results
    print("Search Results:")
    for i, distance in zip(I[0], D[0]):
        print(f"Distance: {distance:.4f}")
        print(f"Title: {data[i]['title']}")
        print(f"URL: {data[i]['url']}")
        print(f"Summary: {data[i]['summery']}")
        print(f"content: {data[i]['content']}")
        print("---")
