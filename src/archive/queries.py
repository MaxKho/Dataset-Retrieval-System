import torch, json
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/galactica-125m")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained("facebook/galactica-125m")
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))
model.to(device)

# Load data
with open("dataset-metadata.json", "r") as f:
    data = json.load(f)

queries = []
pairs = []

BATCH_SIZE = 16
MAX_KEYPHRASE_TOKENS = 150
MAX_QUERY_TOKENS = 50

# Step 1: Generate keyphrases in batches
def batch_generate(prompts, max_tokens):
    inputs = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt", max_length=1024).to(device)
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=0.7
    )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

# Prepare all keyphrase prompts
keyphrase_prompts = [
    f"""Given the dataset description below, extract the following five keyphrases in a structured format:
    1. Tasks mentioned
    2. Scientific domain
    3. Modality of data
    4. Language of data or labels
    5. Rough length of text

    ### Example:

    Description:
    This dataset contains medical radiology reports annotated for anomaly detection in X-rays. The data is labeled in English and spans multiple hospitals in the US. Each sample includes a report paragraph and associated metadata.

    Keyphrases:
    Tasks mentioned: anomaly detection
    Scientific domain: medical imaging
    Modality of data: X-ray images and radiology reports
    Language of data or labels: English
    Rough length of text: short paragraphs

    ### Now for the following dataset:

    Description:
    {example['details']}

    Keyphrases:"""
    for example in data
]

# Run keyphrase generation in batches
keyphrases_list = []
for i in tqdm(range(0, len(keyphrase_prompts), BATCH_SIZE), desc="Generating keyphrases"):
    batch_prompts = keyphrase_prompts[i:i+BATCH_SIZE]
    keyphrases_batch = batch_generate(batch_prompts, MAX_KEYPHRASE_TOKENS)
    keyphrases_list.extend(keyphrases_batch)

# Step 2: Generate researcher queries using keyphrases
query_prompts = [
    f"""You are an academic researcher searching for a dataset. Based on the following structured keyphrases, generate a short and realistic search query you might use to find this dataset.

    ### Example:

    Keyphrases:
    Tasks mentioned: image classification
    Scientific domain: biology
    Modality of data: microscope images
    Language of data or labels: English
    Rough length of text: short captions

    Query:
    Microscope image dataset for biological image classification in English

    ### Now for this set of keyphrases:

    {keyphrases.strip()}

    Query:"""
    for keyphrases in keyphrases_list
]

# Run query generation in batches
query_list = []
for i in tqdm(range(0, len(query_prompts), BATCH_SIZE), desc="Generating queries"):
    batch_prompts = query_prompts[i:i+BATCH_SIZE]
    queries_batch = batch_generate(batch_prompts, MAX_QUERY_TOKENS)
    query_list.extend(queries_batch)

# Construct final dataset with queries
for example, query in zip(data, query_list):
    example["query"] = query.strip()
    queries.append(query.strip())
    pairs.append(example)

# Save output files
with open("queries.json", "w") as f:
    json.dump(queries, f, indent=2)

with open("pairs.json", "w") as f:
    json.dump(pairs, f, indent=2)
