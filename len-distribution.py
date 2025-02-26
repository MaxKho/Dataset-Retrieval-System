import json
import matplotlib.pyplot as plt
import seaborn as sns

data = json.load(open('dataset_descriptions.json'))

# Read and parse the JSON data
descriptions = [item['description'] for item in data]
lengths = [len(desc) for desc in descriptions]

# Set style - using a valid style name
plt.style.use('seaborn-v0_8')  # or simply remove this line as seaborn is already imported

# Create figure and plot
fig, ax = plt.subplots(figsize=(12, 6))

# Create histogram
sns.histplot(data=lengths, bins=50, ax=ax)

# Customize plot
plt.title('Distribution of Dataset Description Lengths', pad=20)
plt.xlabel('Description Length (characters)')
plt.ylabel('Count')

# Add summary statistics as text
stats_text = f'Mean: {int(sum(lengths)/len(lengths)):,}\n'
stats_text += f'Max: {max(lengths):,}\n'
stats_text += f'Min: {min(lengths):,}'
plt.text(0.95, 0.95, stats_text,
         transform=ax.transAxes,
         verticalalignment='top',
         horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Adjust layout and display
plt.tight_layout()
plt.show()