import openml
import pandas as pd

# Initialize the OpenML API with a valid API key
# You can get your API key by registering at https://www.openml.org/
openml.config.apikey = 'YOUR_API_KEY_HERE'  # Replace with your actual API key

try:
    # Get the iris dataset (ID: 61)
    dataset = openml.datasets.get_dataset(61)
    
    # Get the actual data
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        target=dataset.default_target_attribute
    )
    
    # Convert to pandas DataFrame for easier handling
    df = pd.DataFrame(X, columns=attribute_names)
    df['target'] = y
    
    # Display basic information
    print(f"Dataset: {dataset.name}")
    print(f"Number of samples: {len(df)}")
    print(f"Number of features: {len(attribute_names)}")
    
    # Display the first few rows of the dataset
    print("\nFirst few rows of the dataset:")
    print(df.head())
    
    # Save the dataset to a CSV file
    output_filename = f"{dataset.name.lower().replace(' ', '_')}.csv"
    df.to_csv(output_filename, index=False)
    print(f"\nDataset saved to: {output_filename}")
    
except openml.exceptions.OpenMLServerException as e:
    print(f"Authentication error: {e}")
    print("Please make sure to:")
    print("1. Register at https://www.openml.org/")
    print("2. Get your API key from your account settings")
    print("3. Replace 'YOUR_API_KEY_HERE' with your actual API key")
