import openml
import json

# Get a list of all datasets
datasets_list = openml.datasets.list_datasets()

# Create a list to store dataset information
dataset_info_list = []

# Iterate through all datasets and store their information
for dataset_id, dataset_info in datasets_list.items():
    try:
        dataset = openml.datasets.get_dataset(dataset_id)
        
        # Validate id, title and description
        if (dataset_id and 
            dataset_info.get('name') and 
            dataset.description and 
            isinstance(dataset.description, str) and
            isinstance(dataset_info['name'], str)):
            
            # Get feature names (schema labels)
            features = dataset.features
            feature_names = [feature.name for feature in features.values()] if features else []
            
            # Get publication date/year
            year = dataset_info.get('date', '')
            if isinstance(year, str) and len(year) >= 4:
                year = year[:4]  # Extract just the year part
            
            # Concatenate description and publication year
            combined_details = f"[DE]: {dataset.description} [PY]: {year}"
            
            dataset_entry = {
                "id": str(dataset_id),
                "title": dataset_info['name'],
                "details": combined_details,
                "feature_names": feature_names
            }
            dataset_info_list.append(dataset_entry)
            print(f"Processed dataset {dataset_id}: {dataset_info['name']}")
        else:
            print(f"Skipping dataset {dataset_id}: Missing or invalid data")
            
    except Exception as e:
        print(f"Error fetching dataset {dataset_id}: {e}")

# Save the data as JSON
with open('dataset_descriptions.json', 'w', encoding='utf-8') as jsonfile:
    json.dump(dataset_info_list, jsonfile, indent=2)
