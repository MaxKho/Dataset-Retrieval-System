import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from adapters import AutoAdapterModel
import datasets

# Define a PyTorch Dataset wrapper for the SciRepEval "search" public data.
class SciRepEvalSearchDataset(Dataset):
    def __init__(self, split="train"):
        self.dataset = datasets.load_dataset("allenai/scirepeval", "search", split=split)
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        query = item["query"]
        doc = item.get("document")
        if doc is None:
            doc = item.get("title", "") + " " + item.get("abstract", "")
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
# Create the dataset and dataloader.
train_dataset = SciRepEvalSearchDataset(split="train")
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
# Set up the optimizer and loss.
optimizer = optim.Adam(model.parameters(), lr=2e-5)
loss_fn = nn.CrossEntropyLoss()
num_epochs = 3
model.train()

for epoch in range(num_epochs):
    print(f"Epoch: {epoch}")
    epoch_loss = 0.0
    for step, (query_enc, doc_enc) in enumerate(train_loader):
        if step >= 1000:
            break  # Limit to ~5000 steps per epoch

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
    
    avg_loss = epoch_loss / (min(len(train_loader), 5000))
    print(f"Epoch {epoch + 1} finished with average loss: {avg_loss:.4f}")

# Optionally, save the fine-tuned adapters.
model.save_adapter("../data/finetuned_adhoc_query_adapter", "adhoc_query")