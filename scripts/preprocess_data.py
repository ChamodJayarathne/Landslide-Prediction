# import os
# import pandas as pd

# # Get the directory containing the script
# script_dir = os.path.dirname(os.path.abspath(__file__))

# # Load raw data
# rainfall = pd.read_csv(os.path.join(script_dir, '../data/raw/rainfall_data.csv'))
# river_level = pd.read_csv(os.path.join(script_dir, '../data/raw/river_level_data.csv'))
# flood_history = pd.read_csv(os.path.join(script_dir, '../data/raw/flood_history.csv'))
# # rainfall = pd.read_csv('../data/raw/rainfall_data.csv')
# # river_level = pd.read_csv('../data/raw/river_level_data.csv')
# # flood_history = pd.read_csv('../data/raw/flood_history.csv')

# # Merge datasets on Date
# merged_data = pd.merge(rainfall, river_level, on='Date')
# merged_data = pd.merge(merged_data, flood_history, on='Date')

# # Save cleaned data
# merged_data.to_csv('../data/processed/flood_data_cleaned.csv', index=False)
# print("Data preprocessed and saved.")


import os
import pandas as pd

# Get absolute paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)

# Define data directories
raw_data_dir = os.path.join(project_dir, 'data', 'raw')
processed_data_dir = os.path.join(project_dir, 'data', 'processed')

# Create processed directory if it doesn't exist
os.makedirs(processed_data_dir, exist_ok=True)

# Define input and output file paths
rainfall_file = os.path.join(raw_data_dir, 'rainfall_data.csv')
river_level_file = os.path.join(raw_data_dir, 'river_level_data.csv')
flood_history_file = os.path.join(raw_data_dir, 'flood_history.csv')
output_file = os.path.join(processed_data_dir, 'flood_data_cleaned.csv')

# Load raw data
rainfall = pd.read_csv(rainfall_file)
river_level = pd.read_csv(river_level_file)
flood_history = pd.read_csv(flood_history_file)

# Merge datasets on Date
merged_data = pd.merge(rainfall, river_level, on='Date')
merged_data = pd.merge(merged_data, flood_history, on='Date')

# Save cleaned data
merged_data.to_csv(output_file, index=False)
print("Data preprocessed and saved.")