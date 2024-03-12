import cohere
import numpy as np
from tqdm import tqdm
from annoy import AnnoyIndex
import json

# Replace with your Cohere API key
co = cohere.Client('PRTTFHeuJ8i576EqJ46u91AD9xFv2qcPGVNmAbUe')

# Load your dataset
with open("output.json", "r") as f:
    data = json.load(f)
    dataset_texts = [item['content'] for item in data]

# Embed the dataset texts
embeddings = []
for text in tqdm(dataset_texts):
    embed = co.embed(texts=[text])
    print(dir(embed))  # Print out the attributes and methods of the Embeddings object
    # Append the embeddings correctly according to the available attributes/methods
    embeddings.append(embed)  # Adjust this line based on the available attributes/methods

# Convert embeddings to numpy array
embeds = np.array(embeddings)

# Create a search index using Annoy
f = embeds.shape[1]  # Number of dimensions in the embeddings
search_index = AnnoyIndex(f)
for i, embedding in enumerate(embeds):
    search_index.add_item(i, embedding)
search_index.build(10)  # Adjust for desired accuracy/speed trade-off

# Example usage: Search for similar items to a query
query = "trafikvecka"
query_embed = co.embed(texts=[query])
similar_item_ids = search_index.get_nns_by_vector(query_embed, 10)  # Adjust this line based on the available attributes/methods
similar_items = [dataset_texts[item_id] for item_id in similar_item_ids]
print("Similar items to query:", similar_items)
