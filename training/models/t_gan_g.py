import torch
from torch import nn

from training import config
from training.utils import init_weights

class Generator(nn.Module):
    def __init__(self, z_size: int, class_num: int, feature_num: int):
        super().__init__()
        self.z_size = z_size
        self.class_num = class_num
        self.feature_num = feature_num

        # Fully conditional: input will be z + class condition
        self.input_size = z_size + class_num

        self.main_model = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2), 

            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2), 
        )

        self.hidden_status: torch.Tensor = None
        self.last_layer = nn.Sequential(
            nn.Linear(32, feature_num),
            nn.Tanh()
        )
        self.apply(init_weights)
    
    def forward(self, z, class_labels) -> torch.Tensor:
        """
        z: latent vector (batch_size, z_size)
        class_labels: one-hot encoded class labels (batch_size, class_num)
        """
        # Concatenate noise and class condition
        x = torch.cat((z, class_labels), dim=1)
        x = self.main_model(x)
        x = self.last_layer(x)
        return x

    def generate_samples(self, num, class_labels) -> torch.Tensor:
        """
        Generate samples given batch size and class labels
        """
        z = torch.randn(num, self.z_size, device=config.device)
        return self.forward(z, class_labels)