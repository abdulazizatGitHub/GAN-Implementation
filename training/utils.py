import random
import logging
import yaml
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from sklearn.datasets import make_classification, make_blobs
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split

import training

def setup_logging(output_dir: str) -> logging.Logger:
    """Setup logging configuration."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('HDSF-GAN')
    logger.setLevel(logging.INFO)
    
    # Create handlers
    log_file = output_dir / 'training.log'
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler()
    
    # Create formatters and add it to handlers
    log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(log_format)
    console_handler.setFormatter(log_format)
    
    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def load_config(config_path: Path) -> None:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Update GAN config
    for key, value in config_dict.get('GAN_config', {}).items():
        setattr(training.config.GAN_config, key, value)
    
    # Update path config
    for key, value in config_dict.get('path_config', {}).items():
        setattr(training.config.path_config, key, Path(value))

def set_random_state(seed: int = None) -> None:
    if seed is None:
        seed = training.config.seed
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def set_daatset_values():
    training.datasets.feature_num = len(training.datasets.tr_samples[0])
    training.datasets.label_num = int(max(training.datasets.tr_labels).item() + 1)


def prepare_datasets(name: str = None):
    if name is not None:
        tr_features = pd.read_csv(training.config.path_config.datasets / name / 'x_train.csv')
        tr_features = torch.tensor(tr_features.values, dtype=torch.float)
        training.datasets.tr_samples = tr_features

        tr_labels = pd.read_csv(training.config.path_config.datasets / name / 'y_train.csv') 
        tr_labels = torch.tensor(tr_labels.values, dtype=torch.float).squeeze().argmax(1).type(torch.LongTensor)
        training.datasets.tr_labels = tr_labels

        te_features =  pd.read_csv(training.config.path_config.datasets / name / 'x_test.csv')
        te_features = torch.tensor(te_features.values, dtype=torch.float)  
        training.datasets.te_samples = te_features

        te_labels = pd.read_csv(training.config.path_config.datasets / name / 'y_test.csv') 
        te_labels = torch.tensor(te_labels.values, dtype=torch.float).squeeze().argmax(1).type(torch.LongTensor)
        training.datasets.te_labels = te_labels

        set_daatset_values()
    else:
        training.datasets.feature_num = 30
        training.datasets.label_num = 5
        samples, labels = make_blobs(1000, n_features=training.datasets.feature_num, centers=training.datasets.label_num)

        # samples, labels = make_classification(
        #     n_samples=1000,
        #     n_features=training.datasets.feature_num,
        #     n_informative=training.datasets.label_num - 2,
        #     n_redundant=0,
        #     n_classes=5,
        #     n_clusters_per_class=2,
        #     weights=[0.5, 0.3, 0.1, 0.05, 0.05],
        # )

        samples = minmax_scale(samples)
        samples = torch.tensor(samples, dtype=torch.float)
        labels = torch.tensor(labels).type(torch.LongTensor)

        temp = train_test_split(
            samples,
            labels,
            test_size=0.1,
        )

        training.datasets.tr_samples, training.datasets.te_samples, training.datasets.tr_labels, training.datasets.te_labels = temp

def transfer_to_binary():
    for idx, item in enumerate(training.datasets.tr_labels):
        if item > 0:
            training.datasets.tr_labels[idx] = 1
    
    for idx, item in enumerate(training.datasets.te_labels):
        if item > 0:
            training.datasets.te_labels[idx] = 1


def turn_on_test_mode():
    training.datasets.tr_samples = training.datasets.tr_samples[:1000]
    training.datasets.tr_labels = training.datasets.tr_labels[:1000]

    training.datasets.te_samples = training.datasets.te_samples[:1000]
    training.datasets.te_labels = training.datasets.te_labels[:1000]


def init_weights(layer: nn.Module):
    if type(layer) == nn.Linear:
        nn.init.normal_(layer.weight, 0.0, 0.02)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)
    elif type(layer) == nn.BatchNorm1d:
        nn.init.normal_(layer.weight, 1.0, 0.02)
        nn.init.constant_(layer.bias, 0)

def one_hot(labels, num_classes):
        return torch.nn.functional.one_hot(labels, num_classes).float()