import json
import requests

headers = {"Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiNmI2NWViOTUtMDQ1Mi00ODAwLWE4Y2UtYTg4MGJlNzExZTg1IiwidHlwZSI6ImFwaV90b2tlbiJ9.f82kIc5jvv5xz8tIoxczWjH6mB_V8GalEOzGgYDg3d4"}

url = "https://api.edenai.run/v2/text/search"
with open("output.json", "r") as file:
    dataset = json.load(file)

# Extract texts from the dataset
texts = [result["content"] for result in dataset]

text_batches = [texts[i:i+96] for i in range(0, len(texts), 96)]

max_score = -1
max_score_result = None

for batch in text_batches:
    payload = {
        "providers": "cohere",
        "texts": batch,
        "query": "Vad är syftet med Globens lokalpolisområdes medborgarlöfte?",  # Modify this query as needed
        "fallback_providers": ""
    }

    response = requests.post(url, json=payload, headers=headers)

    result = json.loads(response.text)
    print(result)
    if 'cohere' in result and 'items' in result['cohere']:
        # Get the document with the maximum score
        max_score_index = max(range(len(result['cohere']['items'])), key=lambda i: result['cohere']['items'][i]['score'])
        max_score_document_index = result['cohere']['items'][max_score_index]['document']
        max_score_in_batch = result['cohere']['items'][max_score_index]['score']
        
        # Check if this batch contains the highest score so far
        if max_score_in_batch > max_score:
            max_score = max_score_in_batch
            max_score_result = result

# Print the result with the maximum score
if max_score_result is not None:
    max_score_index = max(range(len(max_score_result['cohere']['items'])), key=lambda i: max_score_result['cohere']['items'][i]['score'])
    max_score_document_index = max_score_result['cohere']['items'][max_score_index]['document']
    max_score_document_content = dataset[max_score_document_index]['content']
    print("Content of the document with maximum score:")
    print(max_score_document_content)
else:
    print("No result found")
