#!/usr/bin/env python3
import json
import sys

def format_author(author):
    """
    Format a single author dict into a full name.
    Joins the first name, any middle names, last name, and suffix if available.
    """
    # Get parts from the author dict, ensuring extra spaces are stripped.
    first = author.get("first", "").strip()
    middle_list = author.get("middle", [])
    middle = " ".join(m.strip() for m in middle_list if m) if middle_list else ""
    last = author.get("last", "").strip()
    suffix = author.get("suffix", "").strip()

    # Combine parts; only include non-empty parts.
    parts = [part for part in [first, middle, last, suffix] if part]
    return " ".join(parts)

def transform_record(record):
    """
    Transform one record from the original format into the new format.
    
    New format has:
      - "id": taken from the original "paper_id"
      - "title": taken from the original "title"
      - "feature_names": always set to "unavailable"
      - "details": a concatenated string with authors, abstract, year, doi, and venue.
    """
    new_record = {}
    
    # Map the ID and title.
    new_record["id"] = record.get("paper_id", "")
    new_record["title"] = record.get("title", "")
    
    # For this entire dataset, feature_names is missing; use "unavailable".
    new_record["feature_names"] = "unavailable"
    
    # Format the authors list.
    authors = record.get("authors", [])
    formatted_authors = ", ".join(format_author(author) for author in authors)
    
    # Extract other fields (using empty strings as defaults).
    abstract = record.get("abstract", "")
    year = record.get("year", "")
    doi = record.get("doi", "")
    venue = record.get("venue", "")
    
    # Build a details string.
    details = (
        f"Authors: {formatted_authors}\n"
        f"Abstract: {abstract}\n"
        f"Year: {year}\n"
        f"DOI: {doi}\n"
        f"Venue: {venue}"
    )
    new_record["details"] = details
    
    return new_record

def main():
    if len(sys.argv) < 3:
        print("Usage: python transform.py input.jsonl output.json")
        sys.exit(1)
    
    input_filename = sys.argv[1]
    output_filename = sys.argv[2]
    transformed_records = []
    
    # Process the JSONL file line by line.
    with open(input_filename, 'r', encoding='utf-8') as infile:
        for line_number, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue  # Skip blank lines.
            try:
                record = json.loads(line)
                new_record = transform_record(record)
                transformed_records.append(new_record)
            except json.JSONDecodeError as je:
                print(f"Error decoding JSON on line {line_number}: {je}")
            except Exception as e:
                print(f"Error processing line {line_number}: {e}")
    
    # Write out the list of transformed records as a JSON array.
    with open(output_filename, 'w', encoding='utf-8') as outfile:
        json.dump(transformed_records, outfile, indent=2)
    
    print(f"Transformation complete! {len(transformed_records)} records written to {output_filename}")

if __name__ == "__main__":
    main()
