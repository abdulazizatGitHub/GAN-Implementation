import context

import torch

from training import Classifier, datasets, utils

import numpy as np
import json
from training.config import path_config

# dataset = 'UNSW_NB15'  # Not needed for UNSW-NB15, handled by datasets/__init__.py
# dataset = 'NSL-KDD'

if __name__ == '__main__':
    utils.set_random_state()
    clf = Classifier('test_0')
    clf.fit(datasets.TrDataset())
    clf.test(datasets.TeDataset())
    print(clf.confusion_matrix)
    print(clf.metrics)

    # --- Save results in data/classifier_output ---
    out_dir = path_config.classifier
    np.save(out_dir / 'classifier_matrices.npy', clf.confusion_matrix)
    with open(out_dir / 'classifier_metrics.json', 'w') as f:
        json.dump(clf.metrics, f, indent=2)