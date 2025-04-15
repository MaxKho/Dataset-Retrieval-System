import torch, json
from tqdm import tqdm
from transformers import AutoTokenizer
from adapters import AutoAdapterModel

# Set up the device.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the base tokenizer.
tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")

# Load the base SPECTER2 model.
model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
model.to(device)

# Load the fine-tuned query adapter from a local directory.
# This assumes you have saved your fine-tuned query adapter at the given path.
model.load_adapter("../data/weights/finetuned_adhoc_query_adapter", load_as="adhoc_query")

# Load the proximity adapter for encoding documents from Hugging Face.
model.load_adapter("allenai/specter2", source="hf", load_as="proximity")
# Ensure the model is moved to the device after loading adapters.
model.to(device)

def score(query: str, document: str) -> float:
    """
    Computes the dot product similarity between the query and document embeddings.

    For the query, the function tokenizes the input and uses the "adhoc_query" adapter to encode it.
    For the document, it tokenizes and encodes using the "proximity" adapter.
    It then extracts the CLS token embeddings (the first token) from both outputs and computes the
    dot product between these embeddings to obtain a similarity score.

    Args:
        query (str): The query string.
        document (str): The document string.

    Returns:
        float: The dot product similarity score.
    """
    # Tokenize the query and document.
    query_tokens = tokenizer(query, return_tensors="pt", padding=True,
                             truncation=True, max_length=128).to(device)
    document_tokens = tokenizer(document, return_tensors="pt", padding=True,
                                truncation=True, max_length=512).to(device)

    # Use the query adapter ("adhoc_query") to encode the query.
    model.set_active_adapters("adhoc_query")
    query_output = model(**query_tokens)
    # Extract the CLS token representation.
    query_embedding = query_output.last_hidden_state[:, 0, :]

    # Use the document adapter ("proximity") to encode the document.
    model.set_active_adapters("proximity")
    document_output = model(**document_tokens)
    # Extract the CLS token representation.
    document_embedding = document_output.last_hidden_state[:, 0, :]

    # Compute the dot product between the query and document embeddings.
    similarity_score = torch.sum(query_embedding * document_embedding, dim=1)

    # Return the similarity score as a scalar (assumes single example per call).
    return similarity_score.item()

# Load data
with open("../data/data/3_final_data/final_valid.json", "r") as f:
    data = json.load(f)

# Dictionaries to cache embeddings (mapping from element id to tensor).
query_embeddings = {}
document_embeddings = {}

# Compute embeddings for every element in the data.
# We use torch.no_grad() so that no gradients are computed, reducing memory overhead.
with torch.no_grad():
    for element in tqdm(data, desc="Computing embeddings"):
        entry_id = element["id"]
        query_text = element["query"]
        # Concatenate the document fields.
        doc_text = "[TITLE]\n\n" + element["title"] + "\n\n[DESCRIPTION]\n\n" + element["details"] + "\n\n[FEATURE SUMMARY]\n\n" + element["feature_summary"]

        # Compute query embedding using the "adhoc_query" adapter.
        query_tokens = tokenizer(query_text, return_tensors="pt", padding=True,
                                 truncation=True, max_length=128).to(device)
        model.set_active_adapters("adhoc_query")
        query_output = model(**query_tokens)
        # Extract the CLS token (assumed to be at index 0).
        q_emb = query_output.last_hidden_state[:, 0, :]
        query_embeddings[entry_id] = q_emb

        # Compute document embedding using the "proximity" adapter.
        doc_tokens = tokenizer(doc_text, return_tensors="pt", padding=True,
                               truncation=True, max_length=512).to(device)
        model.set_active_adapters("proximity")
        doc_output = model(**doc_tokens)
        d_emb = doc_output.last_hidden_state[:, 0, :]
        document_embeddings[entry_id] = d_emb

# Now, rank each document for every query using the cached embeddings.
# The ranking is stored in a dictionary where each key (query id)
# is mapped to a list of document ids ranked by similarity (highest first).
ranking = {}

for qid, q_emb in tqdm(query_embeddings.items(), desc="Ranking documents", unit="query"):
    scores = {}
    # Compute dot product similarity between the query embedding and every document embedding.
    for did, d_emb in document_embeddings.items():
        sim_score = torch.sum(q_emb * d_emb).item()
        scores[did] = sim_score
    # Sort document ids based on similarity scores in descending order.
    ranked_doc_ids = sorted(scores, key=scores.get, reverse=True)
    ranking[qid] = ranked_doc_ids

# Save rankings
with open("rankings_valid_concat.json", "w", encoding="utf-8") as f:
    json.dump(ranking, f, indent=4)