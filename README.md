# information-retrieval
Information Retrieval Assignment 2

## Document Creation Script

This repository contains a script (`document-creation.py`) that collects dataset information from OpenML and saves it as a JSON file.

### Prerequisites

Before running the script, you need to:

1. Have Python installed (Python 3.6 or higher recommended)
2. Install the required packages

### Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/information-retrieval.git
   cd information-retrieval
   ```

2. Install the required packages:
   ```
   pip install openml
   ```

### Running the Script

To run the document creation script:

1. Open a terminal or command prompt
2. Navigate to the repository directory
3. Run the script:
   ```
   python document-creation.py
   ```

### What the Script Does

The script:
- Connects to OpenML to retrieve dataset information
- Filters datasets with valid ID, title, and description
- Extracts feature names, publication year, and other metadata
- Combines the information into a structured format
- Saves the collected data to `dataset_descriptions.json`

### Output

After running the script, you'll find a file named `dataset_descriptions.json` in the same directory. This file contains structured information about OpenML datasets including:
- Dataset ID
- Title
- Details (description and publication year)
- Feature names

### Troubleshooting

- If you encounter connection issues, check your internet connection
- For "module not found" errors, ensure you've installed all required packages
- The script may take some time to run as it processes many datasets

