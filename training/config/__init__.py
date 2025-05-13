import torch


from . import (
    c_config,
    GAN_config,
    log_config,
    path_config
)

seed = 0

device: str = 'auto'

if device == 'auto':
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'