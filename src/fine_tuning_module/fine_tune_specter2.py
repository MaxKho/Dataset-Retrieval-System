import json
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from adapters import AutoAdapterModel

# Define a PyTorch Dataset wrapper for our training data stored in a JSON file.
class TrainingDataset(Dataset):
    def init(self, json_path="../data/data/3_final_data/final_train.json"):
        # Load the JSON file containing the training data.
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
    
    def len(self):
        return len(self.data)
    
    def getitem(self, idx):
        item = self.data[idx]
        query = item["query"]
        # Construct the document text by combining title, details, and feature summary.
        title = item.get("title", "")
        details = item.get("details", "")
        feature_summary = item.get("feature_summary", "")
        doc = "[TITLE]\n\n" + title + "\n\n[DESCRIPTION]\n\n" + details + "\n\n[FEATURE SUMMARY]\n\n" + feature_summary
        return query, doc


# Collate function for batching.
def collate_fn(batch):
    queries, docs = zip(*batch)
    query_enc = tokenizer(list(queries), padding=True, truncation=True, return_tensors="pt", max_length=128)
    doc_enc = tokenizer(list(docs), padding=True, truncation=True, return_tensors="pt", max_length=512)
    return query_enc, doc_enc

# Specify the device.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the base tokenizer.
tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")

# Initialize the SPECTER2 base model.
model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
# Move the model to GPU initially.
model.to(device)

# Load the Ad-Hoc Query adapter for encoding short queries.
model.load_adapter("allenai/specter2_adhoc_query", source="hf", load_as="adhoc_query", set_active=True)
# Load the Proximity adapter for encoding candidate documents.
model.load_adapter("allenai/specter2", source="hf", load_as="proximity")
# IMPORTANT: Move the entire model to GPU again after loading adapters.
model.to(device)

# Create the dataset and dataloader using our JSON file.
train_dataset = TrainingDataset(json_path="../data/data/3_final_data/final_train.json")
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

# Set up the optimizer and loss.
optimizer = optim.Adam(model.parameters(), lr=2e-5)
loss_fn = nn.CrossEntropyLoss()
num_epochs = 3
model.train()

for epoch in range(num_epochs):
    print(f"Epoch: {epoch + 1}")
    epoch_loss = 0.0
    for step, (query_enc, doc_enc) in enumerate(train_loader):
        # useful, if dataset is too large!
        # if step >= 1000:
            # break  # Limit to ~1000 steps per epoch

        optimizer.zero_grad()

        # Move input data to the same device as the model.
        query_enc = {k: v.to(device) for k, v in query_enc.items()}
        doc_enc = {k: v.to(device) for k, v in doc_enc.items()}

        # Set active adapter to adhoc_query and encode the queries.
        model.set_active_adapters("adhoc_query")
        query_outputs = model(**query_enc)
        query_repr = query_outputs.last_hidden_state[:, 0, :]  # CLS token embeddings

        # Switch to the proximity adapter and encode the documents.
        model.set_active_adapters("proximity")
        doc_outputs = model(**doc_enc)
        doc_repr = doc_outputs.last_hidden_state[:, 0, :]

        # Compute similarity matrix between queries and documents.
        sim_matrix = torch.matmul(query_repr, doc_repr.T)
        targets = torch.arange(sim_matrix.size(0), device=device)

        loss = loss_fn(sim_matrix, targets)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        if (step + 1) % 50 == 0:
            print(f"Epoch {epoch + 1} | Step {step + 1} | Loss: {loss.item():.4f}")
    
    avg_loss = epoch_loss / (min(len(train_loader), 1000))
    print(f"Epoch {epoch + 1} finished with average loss: {avg_loss:.4f}")

# Optionally, save the fine-tuned adapter.
model.save_adapter("./finetuned_adhoc_query_adapter", "adhoc_query")