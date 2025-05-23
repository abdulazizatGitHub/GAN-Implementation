import pandas as pd
import numpy as np

# Read the training dataset
train_df = pd.read_csv('data/datasets/UNSW_NB15_training-set.csv')

# Filter for specific attack categories
target_categories = ['Normal', 'DoS', 'Reconnaissance', 'Shellcode', 'Worms']
filtered_df = train_df[train_df['attack_cat'].isin(target_categories)]

# Create a mapping for labels
category_mapping = {
    'Normal': 0,
    'DoS': 1,
    'Reconnaissance': 2,
    'Shellcode': 3,
    'Worms': 4
}

# Map the categories to new labels
filtered_df['new_label'] = filtered_df['attack_cat'].map(category_mapping)

# Print basic information
print("\nDataset Shape:", filtered_df.shape)
print("\nColumns:", filtered_df.columns.tolist())
print("\nFirst 5 rows:")
print(filtered_df[['attack_cat', 'new_label']].head())
print("\nData Types:")
print(filtered_df.dtypes)
print("\nMissing Values:")
print(filtered_df.isnull().sum())
print("\nAttack Category Distribution:")
print(filtered_df['attack_cat'].value_counts())
print("\nNew Label Distribution:")
print(filtered_df['new_label'].value_counts())

# Save the filtered dataset
filtered_df.to_csv('data/datasets/UNSW_NB15_filtered_training-set.csv', index=False)
print("\nFiltered dataset saved to 'data/datasets/UNSW_NB15_filtered_training-set.csv'") 