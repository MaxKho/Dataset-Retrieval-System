import json
from rank_bm25 import BM25Okapi

# Helper function for simple tokenization.
def tokenize(text):
    # Lowercase and split on whitespace.
    return text.lower().split()

# Load your custom dataset from a JSON file.
# The JSON file should contain a list of documents in the structure you provided.
with open("documents.json", "r", encoding="utf-8") as f:
    documents = json.load(f)

# Preprocess each document into a single text field.
# Here we concatenate the title, the feature names (joined as a string), and the details.
corpus = []
doc_mapping = []  # To keep track of original document info
for doc in documents:
    # Build a full document string; customize this as you prefer.
    feature_text = " ".join(doc.get("feature_names", []))
    text = f"{doc.get('title', '')} {feature_text} {doc.get('details', '')}"
    corpus.append(tokenize(text))
    doc_mapping.append({
        "id": doc.get("id"),
        "title": doc.get("title"),
        "text": text
    })

# Initialize BM25 with the tokenized corpus.
bm25 = BM25Okapi(corpus)

# Define your custom query.
custom_query = "heat treatment process for improving ductility and reducing hardness in steel"
# Tokenize the query.
tokenized_query = tokenize(custom_query)

# Get BM25 scores for the custom query over all documents.
scores = bm25.get_scores(tokenized_query)

# Get index of the highest scoring document (positive) and lowest scoring document (negative)
pos_idx = scores.argmax()
neg_idx = scores.argmin()

# Retrieve the corresponding documents.
positive_doc = doc_mapping[pos_idx]
negative_doc = doc_mapping[neg_idx]

print("Custom Query:", custom_query)
print("\nPositive Example (Highest BM25 score):")
print(f"ID: {positive_doc['id']}")
print(f"Title: {positive_doc['title']}")
print(f"Score: {scores[pos_idx]:.4f}")
print("Text:", positive_doc["text"][:500], "...\n")  # print first 500 characters

print("Negative Example (Lowest BM25 score):")
print(f"ID: {negative_doc['id']}")
print(f"Title: {negative_doc['title']}")
print(f"Score: {scores[neg_idx]:.4f}")
print("Text:", negative_doc["text"][:500], "...\n")
