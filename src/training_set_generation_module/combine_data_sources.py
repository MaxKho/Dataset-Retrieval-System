import json, sys
from random import sample

def format_author(author):
    """
    Format a single author dict into a full name.
    Joins the first name, any middle names, last name, and suffix if available.
    """
    # Get parts from the author dict, ensuring extra spaces are stripped
    first = author.get("first", "").strip()
    middle_list = author.get("middle", [])
    middle = " ".join(m.strip() for m in middle_list if m) if middle_list else ""
    last = author.get("last", "").strip()
    suffix = author.get("suffix", "").strip()

    # Combine parts; only include non-empty parts
    parts = [part for part in [first, middle, last, suffix] if part]
    return " ".join(parts)

def build_title_to_id_map(original_transformed_raw):
    """
    Build a lookup from each dataset title (in 'positives') to the corresponding paper_id.
    Assumes that each dataset title is unique across all 'positives'.
    """
    title_to_id = {}
    for record in original_transformed_raw:
        paper_id = record.get("paper_id", "")
        for pos_title in record.get("positives", []):
            title_to_id[pos_title.strip()] = paper_id
    return title_to_id

def transform_record(record, title_to_id, all_ids):
    """
    Transform one record from the original format into the new format.

    New format has:
      - "id": taken from the original "paper_id" and prefixed with "0" to avoid clash
      - "title": taken from the original "title"
      - "details": a concatenated string with authors, abstract, year, doi, and venue
      - "feature_summary": always set to [UNAVAILABLE]
      - "query": taken from the original "query"
      - "hard_negatives": taken from the original "negatives" but mapped to IDs using title_to_id
    """
    new_record = {}

    # Map the title and queries
    new_record["title"] = record.get("title", "")
    new_record["query"] = record.get("query", "")

    # For this entire dataset, feature_summary is missing; use [UNAVAILABLE]
    new_record["feature_summary"] = "[UNAVAILABLE]"

    # Build ID with prefix to avoid clash
    original_id = record.get("paper_id", "")
    new_record["id"] = "0" + original_id

    # Convert 'negatives' (titles) into IDs
    negative_titles = record.get("negatives", [])
    hard_negatives = []
    for title in negative_titles:
        mapped_id = title_to_id.get(title.strip())
        if mapped_id:
            hard_negatives.append("0" + mapped_id)  # also prefix these IDs

    if not hard_negatives:
        candidates = [doc_id for doc_id in all_ids if doc_id != "0" + original_id]
        hard_negatives = sample(candidates, min(5, len(candidates)))

    new_record["hard_negatives"] = hard_negatives

    # Format the authors list
    authors = record.get("authors", [])
    formatted_authors = ", ".join(format_author(author) for author in authors)

    # Extract other fields (using empty strings as defaults)
    abstract = record.get("abstract", "")
    year = record.get("year", "")
    doi = record.get("doi", "")
    venue = record.get("venue", "")

    # Build a details string
    details = (
        f"Authors: {formatted_authors}\n"
        f"Abstract: {abstract}\n"
        f"Year: {year}\n"
        f"DOI: {doi}\n"
        f"Venue: {venue}"
    )
    new_record["details"] = details

    return new_record

# Transform the original query-document pairs and combine it with our pairs and save as json file
with open("../data/data/2_intermediate_data/train_data.jsonl", "r") as f:
    original = [json.loads(line) for line in f]
with open("../data/data/2_intermediate_data/pairs_with_negatives.json", "r") as f:
    pairs = json.load(f)

title_to_id = build_title_to_id_map(original)
all_ids = ["0" + record["paper_id"] for record in original]
original_transformed = [transform_record(record, title_to_id, all_ids) for record in original]
combined = original_transformed + pairs

with open("../data/data/2_intermediate_data/training_data.json", "w") as f:
    json.dump(combined, f, indent=2)

print("✅ training_data.json file with summarised features created.")
