import json
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from adapters import AutoAdapterModel

# Define a PyTorch Dataset wrapper for our training data stored in a JSON file.
# Note: we use __init__, __len__, __getitem__ so that our dataset works correctly.
class TrainingDataset(Dataset):
    def __init__(self, json_path="../data/data/3_final_data/final_train.json", doc_field="title"):
        # Load the JSON file containing the training data.
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.doc_field = doc_field
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        query = item["query"]
        # Use only the selected document field (either "title", "details", or "feature_summary")
        doc = item.get(self.doc_field, "")
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

# We will loop over three different document fields.
doc_fields = ["title", "details", "feature_summary"]

for field in doc_fields:
    print(f"\n=== Training with document field: {field} ===")

    # Initialize the SPECTER2 base model for each run.
    model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
    model.to(device)
    
    # Load the Ad-Hoc Query adapter for encoding short queries.
    model.load_adapter("allenai/specter2_adhoc_query", source="hf", load_as="adhoc_query", set_active=True)
    # Load the Proximity adapter for encoding candidate documents.
    model.load_adapter("allenai/specter2", source="hf", load_as="proximity")
    model.to(device)

    # Create the dataset and dataloader using our JSON file and the selected document field.
    train_dataset = TrainingDataset(json_path="../data/data/3_final_data/final_train.json", doc_field=field)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

    # Set up the optimizer and loss.
    optimizer = optim.Adam(model.parameters(), lr=2e-5)
    loss_fn = nn.CrossEntropyLoss()
    num_epochs = 20
    model.train()

    for epoch in range(num_epochs):
        print(f"Epoch: {epoch + 1} (doc_field: {field})")
        epoch_loss = 0.0
        for step, (query_enc, doc_enc) in enumerate(train_loader):
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
                print(f"Field: {field} | Epoch {epoch + 1} | Step {step + 1} | Loss: {loss.item():.4f}")
        
        avg_loss = epoch_loss / (min(len(train_loader), 1000))
        print(f"Field: {field} | Epoch {epoch + 1} finished with average loss: {avg_loss:.4f}")
    
    # Save the fine-tuned adapter with a unique name using the field.
    save_path = f"../data/weights/finetuned_adhoc_query_adapter_{field}"
    model.save_adapter(save_path, "adhoc_query")
    print(f"Training with document field '{field}' complete. Model saved to {save_path}\n")