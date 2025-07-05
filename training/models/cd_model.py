import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm

from training.utils import init_weights

class CDModel(nn.Module):
    def __init__(self, in_features: int, label_num: int):
        super().__init__()
        # Shared feature extraction body
        self.main_model = nn.Sequential(
            spectral_norm(nn.Linear(in_features, 1024)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(1024, 512)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(512, 128)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(128, 32)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(32, 16)),
            nn.LeakyReLU(0.2),
        )
        
        # Head 1: Critic for WGAN loss (outputs a single "realness" score)
        self.d_last_layer = nn.Sequential(
            nn.Linear(16, 1)
        )

        # Head 2: Classifier for classification loss (outputs logits for each class)
        self.c_last_layer = nn.Sequential(
            nn.Linear(16, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(64, label_num)
        )

        self.apply(init_weights)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.main_model(x)
        critic_score = self.d_last_layer(features)
        class_logits = self.c_last_layer(features)
        return critic_score, class_logits