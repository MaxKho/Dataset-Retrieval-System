import openml
import pandas as pd
from collections import Counter
import re
from tqdm import tqdm  # for progress bar

try:
    # Get list of all datasets (metadata only)
    print("Fetching dataset metadata...")
    datasets = openml.datasets.list_datasets(output_format="dataframe")
    
    # Create a list to store detailed information
    detailed_info = []
    
    # Get detailed information for first 10 datasets (reduced for testing)
    print("\nFetching detailed information for datasets...")
    for index, row in tqdm(datasets.head(10).iterrows()):
        try:
            did = row['did']
            print(f"\nProcessing dataset {did}: {row['name']}")
            
            # Get detailed dataset information
            dataset = openml.datasets.get_dataset(did, download_data=False)
            
            # Extract detailed metadata with error checking
            info = {
                'did': did,
                'name': str(dataset.name),
                'description': str(dataset.description) if dataset.description else "No description",
                'creator': str(dataset.creator) if dataset.creator else "Unknown",
                'tags': ', '.join(dataset.tag) if dataset.tag else "",
                'licence': str(dataset.licence) if dataset.licence else "Unknown",
                'version': str(dataset.version),
                'number_instances': dataset.number_instances,
                'number_features': dataset.number_features,
                'number_missing_values': dataset.number_missing_values,
                'url': f"https://www.openml.org/d/{did}"
            }
            
            # Optional fields - add if they exist
            if hasattr(dataset, 'paper_url'):
                info['paper_url'] = dataset.paper_url
            if hasattr(dataset, 'original_data_url'):
                info['original_data_url'] = dataset.original_data_url
            if hasattr(dataset, 'collection_date'):
                info['collection_date'] = dataset.collection_date
            if hasattr(dataset, 'language'):
                info['language'] = dataset.language
            
            detailed_info.append(info)
            print(f"Successfully processed dataset {did}")
            
        except Exception as e:
            print(f"\nError processing dataset {did}: {str(e)}")
            continue
    
    if not detailed_info:
        print("\nNo data was collected! Check if the OpenML connection is working.")
    else:
        # Convert to DataFrame
        detailed_df = pd.DataFrame(detailed_info)
        
        # Save detailed information
        detailed_df.to_csv('openml_datasets_detailed.csv', index=False)
        print(f"\nDetailed information saved for {len(detailed_info)} datasets")
        
        # Show example of collected data
        print("\nExample of collected data:")
        print(detailed_df.head(1).to_string())

except openml.exceptions.OpenMLServerException as e:
    print(f"Error accessing OpenML server: {e}")
except Exception as e:
    print(f"An error occurred: {e}")

# List all datasets and their properties
openml.datasets.list_datasets(output_format="dataframe")

# Get dataset by name
dataset = openml.datasets.get_dataset('Fashion-MNIST')

print(dataset)