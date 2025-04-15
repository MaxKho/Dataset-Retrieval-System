import sys
import json
import torch
import pickle
from transformers import AutoTokenizer
from adapters import AutoAdapterModel

def run_partial(query_input: str, top_k: int = 10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load precomputed document embeddings (assumed to be saved as a pickle file).
    with open("../data/data/3_final_data/doc_embeddings_partial.pkl", "rb") as f:
        doc_embeddings = pickle.load(f)
    
    # Load the document data (for printing metadata later).
    with open("../data/data/3_final_data/final_test.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Load model and tokenizer (only needed to compute the query embeddings).
    tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
    model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
    model.to(device)
    model.load_adapter("../data/weights/finetuned_adhoc_query_adapter_title", load_as="title")
    model.load_adapter("../data/weights/finetuned_adhoc_query_adapter_details", load_as="details")
    model.load_adapter("../data/weights/finetuned_adhoc_query_adapter_feature_summary", load_as="feature_summary")
    model.to(device)
    model.eval()
    
    # Compute the query embeddings (one per adapter).
    query_embeddings = {}
    for adapter, max_length in zip(["title", "details", "feature_summary"], [128, 128, 128]):
        tokens = tokenizer(query_input, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        tokens = {k: v.to(device) for k, v in tokens.items()}
        model.set_active_adapters(adapter)
        out = model(**tokens)
        # Move each query embedding to CPU so that it's on the same device as the stored embeddings.
        query_embeddings[adapter] = out.last_hidden_state[:, 0, :].squeeze(0).cpu()
    
    # Compute composite similarity scores.
    scores = {}
    for eid, vals in doc_embeddings.items():
        rec = vals["record"]
        # Choose weights based on availability of the feature summary.
        if rec["feature_summary"] == "[UNAVAILABLE]":
            w_title, w_details, w_feature = 0.1, 0.9, 0.0
        else:
            w_title, w_details, w_feature = 0.1, 0.6, 0.3
        
        sim_title = torch.dot(query_embeddings["title"], vals["emb_title"]).item()
        sim_details = torch.dot(query_embeddings["details"], vals["emb_details"]).item()
        sim_feature = torch.dot(query_embeddings["feature_summary"], vals["emb_feature"]).item()
        total_score = w_title * sim_title + w_details * sim_details + w_feature * sim_feature
        scores[eid] = total_score
    
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    print(f"Top {top_k} results for query: '{query_input}' (partial model):")
    for eid, score_val in ranked[:top_k]:
        record = next(item for item in data if item["id"] == eid)
        print(f"ID: {record['id']}, Title: {record['title']}, Score: {score_val}")

def run_joint(query_input: str, top_k: int = 10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load precomputed document embeddings.
    with open("../data/data/3_final_data/doc_embeddings_joint.pkl", "rb") as f:
        doc_embeddings = pickle.load(f)
    
    with open("../data/data/3_final_data/final_test.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
    model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
    model.to(device)
    model.load_adapter("../data/weights/finetuned_adhoc_query_adapter", load_as="adhoc_query")
    model.load_adapter("allenai/specter2", source="hf", load_as="proximity")
    model.to(device)
    model.eval()
    
    # Compute the query embedding.
    tokens = tokenizer(query_input, return_tensors="pt", padding=True, truncation=True, max_length=128)
    tokens = {k: v.to(device) for k, v in tokens.items()}
    model.set_active_adapters("adhoc_query")
    out = model(**tokens)
    # Move the query embedding to CPU.
    query_emb = out.last_hidden_state[:, 0, :].squeeze(0).cpu()
    
    scores = {}
    for eid, vals in doc_embeddings.items():
        d_emb = vals["embedding"]
        score_val = torch.sum(query_emb * d_emb).item()
        scores[eid] = score_val
    
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    print(f"Top {top_k} results for query: '{query_input}' (joint model):")
    for eid, score_val in ranked[:top_k]:
        record = next(item for item in data if item["id"] == eid)
        print(f"ID: {record['id']}, Title: {record['title']}, Score: {score_val}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python demo_retrieval.py [joint|partial] \"query\" [top_k (default=10)]")
        sys.exit(1)

    mode = sys.argv[1].lower()
    query = sys.argv[2]
    
    # If a third argument is provided, use it for top_k; otherwise, default to 10.
    if len(sys.argv) >= 4:
        try:
            top_k = int(sys.argv[3])
        except ValueError:
            print("Invalid top_k value; please provide an integer.")
            sys.exit(1)
    else:
        top_k = 10

    if mode == "partial":
        run_partial(query, top_k)
    elif mode == "joint":
        run_joint(query, top_k)
    else:
        print("Invalid mode. Use 'joint' or 'partial'.")