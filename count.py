import json

def count_entries(json_data):
    # Parse the JSON string into a list of dictionaries
    entries = json.loads(json_data)
    
    # Return the length of the list
    return len(entries)

# Example usage:
with open('dataset_descriptions.json', 'r') as file:
    json_data = file.read()
    count = count_entries(json_data)
    print(f"Number of entries: {count}")
