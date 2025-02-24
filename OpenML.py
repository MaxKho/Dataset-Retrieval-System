import openml
import pandas as pd

try:
    # Get list of all datasets (metadata only)
    print("Fetching dataset metadata...")
    datasets = openml.datasets.list_datasets(output_format="dataframe")
    
    # Display basic information about available datasets
    print("\nAvailable Datasets:")
    print(f"Total number of datasets: {len(datasets)}")
    
    # Print available columns
    print("\nAvailable columns in the dataset:")
    print(datasets.columns.tolist())
    
    # Show first 10 datasets with basic information
    print("\nFirst 10 datasets:")
    # Using more commonly available columns
    print(datasets[['did', 'name', 'NumberOfInstances', 'NumberOfFeatures']].head(10))
    
    # Save metadata to CSV for easier browsing
    datasets.to_csv('openml_datasets.csv')
    print("\nFull dataset list saved to 'openml_datasets.csv'")

    # Example: Download an actual dataset (using iris as an example)
    print("\nDownloading an example dataset (iris)...")
    dataset = openml.datasets.get_dataset(61)  # 61 is the ID for iris
    X, y, categorical_indicator, attribute_names = dataset.get_data(dataset_format="dataframe")
    
    print("\nExample dataset (iris) details:")
    print(f"Shape of features (X): {X.shape}")
    print(f"Shape of target (y): {y.shape}")
    print("\nFirst few rows:")
    print(X.head())

except openml.exceptions.OpenMLServerException as e:
    print(f"Error accessing OpenML server: {e}")
except Exception as e:
    print(f"An error occurred: {e}")

# List all datasets and their properties
openml.datasets.list_datasets(output_format="dataframe")

# Get dataset by name
dataset = openml.datasets.get_dataset('Fashion-MNIST')

print(dataset)