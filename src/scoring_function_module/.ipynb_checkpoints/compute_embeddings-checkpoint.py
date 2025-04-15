import sys
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from adapters import AutoAdapterModel
import pickle

def create_doc_embeddings_partial(data_file, output_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
    model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
    model.to(device)
    
    # Load the three adapters for the partial model.
    model.load_adapter("../data/weights/finetuned_adhoc_query_adapter_title", load_as="title")
    model.load_adapter("../data/weights/finetuned_adhoc_query_adapter_details", load_as="details")
    model.load_adapter("../data/weights/finetuned_adhoc_query_adapter_feature_summary", load_as="feature_summary")
    # Ensure all adapter parameters are on the device.
    model.to(device)
    model.eval()
    
    # Load document data.
    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    doc_embeddings = {}
    with torch.no_grad():
        for element in tqdm(data, desc="Computing document embeddings (partial)"):
            eid = element["id"]
            
            # --- Title ---
            tokens = tokenizer(element["title"], return_tensors="pt", padding=True,
                               truncation=True, max_length=128)
            tokens = {k: v.to(device) for k, v in tokens.items()}
            model.set_active_adapters("title")
            out = model(**tokens)
            emb_title = out.last_hidden_state[:, 0, :].squeeze(0).cpu()
            
            # --- Details ---
            tokens = tokenizer(element["details"], return_tensors="pt", padding=True,
                               truncation=True, max_length=512)
            tokens = {k: v.to(device) for k, v in tokens.items()}
            model.set_active_adapters("details")
            out = model(**tokens)
            emb_details = out.last_hidden_state[:, 0, :].squeeze(0).cpu()
            
            # --- Feature Summary ---
            tokens = tokenizer(element["feature_summary"], return_tensors="pt", padding=True,
                               truncation=True, max_length=512)
            tokens = {k: v.to(device) for k, v in tokens.items()}
            model.set_active_adapters("feature_summary")
            out = model(**tokens)
            emb_feature = out.last_hidden_state[:, 0, :].squeeze(0).cpu()
            
            doc_embeddings[eid] = {
                "emb_title": emb_title,
                "emb_details": emb_details,
                "emb_feature": emb_feature,
                "record": element
            }
    
    with open(output_file, "wb") as f:
        pickle.dump(doc_embeddings, f)
    print(f"Document embeddings (partial) saved to {output_file}")

def create_doc_embeddings_joint(data_file, output_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
    model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
    model.to(device)
    
    # Load the adapters for the joint model.
    model.load_adapter("../data/weights/finetuned_adhoc_query_adapter", load_as="adhoc_query")
    model.load_adapter("allenai/specter2", source="hf", load_as="proximity")
    model.to(device)
    model.eval()
    
    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    doc_embeddings = {}
    with torch.no_grad():
        for element in tqdm(data, desc="Computing document embeddings (joint)"):
            eid = element["id"]
            # Concatenate document fields.
            doc_text = (
                "[TITLE]\n\n" + element["title"] +
                "\n\n[DESCRIPTION]\n\n" + element["details"] +
                "\n\n[FEATURE SUMMARY]\n\n" + element["feature_summary"]
            )
            tokens = tokenizer(doc_text, return_tensors="pt", padding=True,
                               truncation=True, max_length=512)
            tokens = {k: v.to(device) for k, v in tokens.items()}
            model.set_active_adapters("proximity")
            out = model(**tokens)
            emb = out.last_hidden_state[:, 0, :].squeeze(0).cpu()
            doc_embeddings[eid] = {"embedding": emb, "record": element}
    
    with open(output_file, "wb") as f:
        pickle.dump(doc_embeddings, f)
    print(f"Document embeddings (joint) saved to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python compute_embeddings.py [partial|joint] <data_file.json> <output_file.pkl>")
        sys.exit(1)
    
    mode = sys.argv[1].lower()
    data_file = sys.argv[2]
    output_file = sys.argv[3]
    
    if mode == "partial":
        create_doc_embeddings_partial(data_file, output_file)
    elif mode == "joint":
        create_doc_embeddings_joint(data_file, output_file)
    else:
        print("Invalid mode specified. Use 'partial' or 'joint'.")