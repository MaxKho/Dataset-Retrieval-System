#!/usr/bin/env python3
import json
import os

def load_data(file_path):
    """
    Load a file that may be either a JSON array (entire file as one JSON object)
    or a JSONL file (one JSON object per line).
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        # Peek at the first non-whitespace character.
        first_char = None
        while True:
            char = f.read(1)
            if not char:
                break
            if not char.isspace():
                first_char = char
                break
        f.seek(0)  # Reset file pointer to the beginning.
        
        if first_char == '[':
            # The file contains a JSON array.
            return json.load(f)
        else:
            # The file is in JSONL format (each line is a valid JSON record).
            records = []
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
            return records

def main():
    # Define input file paths relative to this script's location.
    # new_train_data.jsonl is stored in main/data/
    jsonl_file_path = os.path.join("..", "data", "new_train_data.jsonl")
    # dataset-metadata.json is now in main/data/ too.
    json_file_path = os.path.join("..", "data", "dataset-metadata.json")
    
    # Define output file path (we'll write the combined data to main/data/combined_data.json)
    output_file_path = os.path.join("..", "data", "combined_data.json")
    
    # Load the datasets
    print(f"Loading data from: {jsonl_file_path}")
    data1 = load_data(jsonl_file_path)
    print(f"Loaded {len(data1)} records from the first data source.")
    
    print(f"Loading data from: {json_file_path}")
    data2 = load_data(json_file_path)
    print(f"Loaded {len(data2)} records from the second data source.")
    
    # Combine the records from both data sources.
    combined_data = data1 + data2
    print(f"Combined total records: {len(combined_data)}")
    
    # Write the combined data to the output file as a JSON array.
    with open(output_file_path, 'w', encoding='utf-8') as outfile:
        json.dump(combined_data, outfile, indent=2, ensure_ascii=False)
    print(f"Combined data written to: {output_file_path}")

if __name__ == '__main__':
    main()
