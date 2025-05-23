import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

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

# Create preprocessing pipeline using MinMaxScaler
preprocessor = ColumnTransformer(
    transformers=[
        ('num', MinMaxScaler(), numerical_columns),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_columns)
    ])

def load_and_preprocess_data(file_path):
    # Read the dataset
    df = pd.read_csv(file_path)
    
    # Filter for specific attack categories
    target_categories = ['Normal', 'DoS', 'Reconnaissance', 'Shellcode', 'Worms']
    df = df[df['attack_cat'].isin(target_categories)]
    
    # Create label mapping
    category_mapping = {
        'Normal': 0,
        'DoS': 1,
        'Reconnaissance': 2,
        'Shellcode': 3,
        'Worms': 4
    }
    
    # Extract features and labels
    X = df.drop(['id', 'attack_cat', 'label'], axis=1)
    y = df['attack_cat'].map(category_mapping)
    
    # Preprocess features
    X_processed = preprocessor.fit_transform(X)
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(X_processed)
    y_tensor = torch.LongTensor(y.values)
    
    return X_tensor, y_tensor

# Load training and test data
tr_samples, tr_labels = load_and_preprocess_data('data/datasets/UNSW_NB15_training-set.csv')
te_samples, te_labels = load_and_preprocess_data('data/datasets/UNSW_NB15_testing-set.csv')

# Update feature and label dimensions
feature_num = tr_samples.shape[1]  # Number of features after preprocessing
label_num = 5  # Number of classes (Normal, DoS, Reconnaissance, Shellcode, Worms)

# Calculate class weights for handling imbalance
class_counts = torch.bincount(tr_labels)
class_weights = 1.0 / class_counts
class_weights = class_weights / class_weights.sum()

pass