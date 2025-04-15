import json
import random

# Set the random seed for reproducibility
seed = 42
random.seed(seed)

# Load the combined data from the JSON file using utf-8 encoding
with open('../data/data/2_intermediate_data/combined_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Shuffle the data to ensure randomness
random.shuffle(data)

# Determine split sizes using a 70/10/20 ratio in your updated code
total = len(data)
train_size = int(total * 0.7)
valid_size = int(total * 0.1)
test_size = total - train_size - valid_size

# Split the data
train_data = data[:train_size]
valid_data = data[train_size:train_size + valid_size]
test_data = data[train_size + valid_size:]

# Save each split into a separate JSON file
with open('../data/data/3_final_data/final_train.json', 'w', encoding='utf-8') as f:
    json.dump(train_data, f, indent=4)

with open('../data/data/3_final_data/final_valid.json', 'w', encoding='utf-8') as f:
    json.dump(valid_data, f, indent=4)

with open('../data/data/3_final_data/final_test.json', 'w', encoding='utf-8') as f:
    json.dump(test_data, f, indent=4)

print(f"Data split complete: {len(train_data)} train, {len(valid_data)} valid, {len(test_data)} test records.")
