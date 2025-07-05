import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif

from training.config import path_config
from training.datasets.tr_dataset import TrDataset
from training.datasets.te_dataset import TeDataset

# Define categorical and numerical columns
categorical_columns = ['proto', 'service', 'state']
numerical_columns = ['dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'sttl', 'dttl', 
                    'sload', 'dload', 'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit',
                    'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt', 'synack', 'ackdat', 'smean',
                    'dmean', 'trans_depth', 'response_body_len', 'ct_srv_src', 'ct_state_ttl',
                    'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm',
                    'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd', 'ct_src_ltm', 'ct_srv_dst',
                    'is_sm_ips_ports']

def load_and_preprocess_data(train_path, test_path):
    # Read datasets
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    # Filter for specific attack categories
    target_categories = ['Normal', 'DoS', 'Reconnaissance', 'Shellcode', 'Worms']
    df_train = df_train[df_train['attack_cat'].isin(target_categories)]
    df_test = df_test[df_test['attack_cat'].isin(target_categories)]

    # Create label mapping
    category_mapping = {
        'Normal': 0,
        'DoS': 1,
        'Reconnaissance': 2,
        'Shellcode': 3,
        'Worms': 4
    }

    # Extract features and labels
    X_train = df_train.drop(['id', 'attack_cat', 'label'], axis=1)
    y_train = df_train['attack_cat'].map(category_mapping)
    X_test = df_test.drop(['id', 'attack_cat', 'label'], axis=1)
    y_test = df_test['attack_cat'].map(category_mapping)

    # Print unique values in categorical columns
    print("\nUnique values in categorical columns:")
    for col in categorical_columns:
        print(f"{col}: {len(X_train[col].unique())} unique values")

    # Fit preprocessor on training data only
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', MinMaxScaler(), numerical_columns),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_columns)
        ]
    )
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Select the top 55 features using SelectKBest
    selector = SelectKBest(f_classif, k=55)
    X_train_processed = selector.fit_transform(X_train_processed, y_train)
    X_test_processed = selector.transform(X_test_processed)

    # Print feature dimensions
    print(f"\nFeature dimensions after preprocessing (SelectKBest to 55):")
    print(f"Total features: {X_train_processed.shape[1]}")

    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train_processed)
    y_train_tensor = torch.LongTensor(y_train.values)
    X_test_tensor = torch.FloatTensor(X_test_processed)
    y_test_tensor = torch.LongTensor(y_test.values)

    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, preprocessor

# Usage:
tr_samples, tr_labels, te_samples, te_labels, preprocessor = load_and_preprocess_data(
    'data/datasets/UNSW_NB15_training-set.csv',
    'data/datasets/UNSW_NB15_testing-set.csv'
)

# Update feature and label dimensions
feature_num = tr_samples.shape[1]  # Number of features after preprocessing
label_num = 5  # Number of classes (Normal, DoS, Reconnaissance, Shellcode, Worms)

# Calculate class weights and imbalance ratios for handling imbalance
class_counts = torch.bincount(tr_labels)
max_count = class_counts.max()
imbalance_ratios = {i: (max_count / count).item() for i, count in enumerate(class_counts)}
class_weights = 1.0 / class_counts
class_weights = class_weights / class_weights.sum()

# Print class distribution information
print("\nClass Distribution:")
for i, (count, ratio) in enumerate(imbalance_ratios.items()):
    print(f"Class {i}: Count = {count}, Imbalance Ratio = {ratio:.2f}")

pass