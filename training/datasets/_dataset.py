import torch


from training import datasets, config


class Dataset:
    def _init_(self, training: bool = True):
        if training:
            self.samples = datasets.tr_samples.to(config.device)
            self.labels = datasets.tr_labels.to(config.device)
        else:
            self.samples = datasets.te_samples.to(config.device)
            self.labels = datasets.te_labels.to(config.device)

    def _len_(self):
        return len(self.labels)

    def _getitem_(self, idx: int):
        return self.samples[idx], self.labels[idx]