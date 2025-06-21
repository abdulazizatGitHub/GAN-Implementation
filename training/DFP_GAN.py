# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.nn.utils import spectral_norm
# import numpy as np
# from typing import List, Dict, Tuple

# from training import config, datasets, models
# from training.TMG_GAN import TMGGAN
# from training.tracker import TrainingTracker
# from scripts.visualize_gan_output import Visualizer

# class AnchorLoss(nn.Module):
#     """Latent Space Triangulation (LST) Loss"""
#     def __init__(self, anchors: torch.Tensor):
#         super().__init__()
#         self.anchors = anchors  # Precomputed per-class mean features [num_classes, feature_dim]
        
#     def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
#         # Align generated features with class anchors
#         anchor_targets = self.anchors[labels]
#         return torch.mean(torch.norm(features - anchor_targets, dim=1))

# def dagp_loss(generator: nn.Module, z1: torch.Tensor, z2: torch.Tensor, gamma: float = 0.1) -> torch.Tensor:
#     """Diversity-Aware Gradient Penalty (DAGP)"""
#     samples1 = generator(z1)
#     samples2 = generator(z2)
#     # Compute L2 distance instead of L1 for better gradient flow
#     pairwise_dist = torch.norm(samples1 - samples2, dim=1)
#     # Use a smoother penalty function
#     return torch.mean(torch.relu(gamma - pairwise_dist) ** 2)

# class DFPGAN(TMGGAN):
#     def __init__(self):
#         super().__init__()
#         self.anchors = None  # Will be initialized during training
#         self.anchor_criterion = None
#         self.gamma = 0.5  # Increased DAGP diversity threshold
#         self.feature_dim = 16  # CD model's hidden feature dimension
#         self.lst_weight = 0.1  # Reduced LST loss weight
#         self.dagp_weight = 0.5  # Increased DAGP loss weight
        
#     def _compute_anchors(self, dataset: datasets.TrDataset) -> torch.Tensor:
#         """Initialize anchors from real data features"""
#         anchors = torch.zeros(datasets.label_num, self.feature_dim, device=config.device)
#         self.cd.eval()  # Set to evaluation mode for feature extraction
        
#         with torch.no_grad():
#             for label in range(datasets.label_num):
#                 if label not in self.samples or len(self.samples[label]) == 0:
#                     continue
                    
#                 # Get real samples for this class
#                 real_samples = self.get_target_samples(label, min(len(self.samples[label]), 1000))
#                 # Extract features using CD model
#                 _, _ = self.cd(real_samples)  # Forward pass to get hidden features
#                 features = self.cd.hidden_status  # Get the hidden features (16-dim)
#                 # Compute mean features as anchor
#                 anchors[label] = features.mean(dim=0)
        
#         self.cd.train()  # Set back to training mode
#         return anchors
    
#     def update_anchors(self, dataset: datasets.TrDataset):
#         """Update anchors via moving average"""
#         with torch.no_grad():
#             new_anchors = self._compute_anchors(dataset)
#             if self.anchors is None:
#                 self.anchors = new_anchors
#             else:
#                 self.anchors = 0.9 * self.anchors + 0.1 * new_anchors  # β=0.9
#             self.anchor_criterion = AnchorLoss(self.anchors)
    
#     def fit(self, dataset: datasets.TrDataset):
#         """DFP-GAN specific training process"""
#         # Initialize models
#         self.cd.train()
#         for i in self.generators:
#             i.train()
        
#         # Divide samples into classes
#         self.divideSamples(dataset)
        
#         # Initialize anchors
#         self.update_anchors(dataset)
        
#         # Initialize optimizers
#         cd_optimizer = torch.optim.Adam(
#             params=self.cd.parameters(),
#             lr=config.GAN_config.cd_lr,
#             betas=(0.5, 0.999)
#         )
        
#         g_optimizers = [
#             torch.optim.Adam(
#                 params=self.generators[i].parameters(),
#                 lr=config.GAN_config.g_lr,
#                 betas=(0.5, 0.999)
#             )
#             for i in range(datasets.label_num)
#         ]
        
#         # Training loop
#         for e in range(config.GAN_config.epochs):
#             print(f'\r{(e + 1) / config.GAN_config.epochs: .2%}', end='')
            
#             epoch_d_loss = 0.0
#             epoch_g_loss = 0.0
#             epoch_c_loss = 0.0
#             batch_count = 0
            
#             for target_label in self.samples.keys():
#                 # Train critic/discriminator
#                 for _ in range(config.GAN_config.cd_loopNo):
#                     cd_optimizer.zero_grad()
                    
#                     # Get real and generated samples
#                     real_samples = self.get_target_samples(target_label, config.GAN_config.batch_size)
#                     generated_samples = self.generators[target_label].generate_samples(config.GAN_config.batch_size)
                    
#                     # Get predictions
#                     score_real, predicted_labels_real = self.cd(real_samples)
#                     score_generated, predicted_labels_gen = self.cd(generated_samples)
                    
#                     # Compute losses
#                     d_loss = (score_generated.mean() - score_real.mean()) / 2
#                     c_loss = F.cross_entropy(
#                         input=predicted_labels_real,
#                         target=torch.full([len(predicted_labels_real)], target_label, device=config.device),
#                         weight=datasets.class_weights.to(config.device) if predicted_labels_real.shape[1] == len(datasets.class_weights) else None
#                     )
                    
#                     # Total loss and update
#                     loss = d_loss + c_loss
#                     loss.backward()
#                     cd_optimizer.step()
                    
#                     epoch_d_loss += d_loss.item()
#                     epoch_c_loss += c_loss.item()
#                     batch_count += 1
                
#                 # Train generator
#                 for _ in range(config.GAN_config.g_loopNo):
#                     g_optimizers[target_label].zero_grad()
                    
#                     # Generate samples
#                     z = torch.randn(config.GAN_config.batch_size, config.GAN_config.z_size, device=config.device)
#                     generated_samples = self.generators[target_label].generate_samples(config.GAN_config.batch_size)
                    
#                     # Get predictions
#                     score_generated, predicted_labels = self.cd(generated_samples)
                    
#                     # Original TMG-GAN losses
#                     score_generated = score_generated.mean()
#                     loss_label = F.cross_entropy(
#                         input=predicted_labels,
#                         target=torch.full([len(predicted_labels)], target_label, device=config.device),
#                         weight=datasets.class_weights.to(config.device) if predicted_labels.shape[1] == len(datasets.class_weights) else None
#                     )
                    
#                     # DFP-GAN specific losses
#                     # 1. Latent Space Triangulation (LST)
#                     gen_features = self.cd.hidden_status
#                     lst_loss = self.anchor_criterion(gen_features, torch.full([len(generated_samples)], target_label, device=config.device))
                    
#                     # 2. Diversity-Aware Gradient Penalty (DAGP)
#                     z1 = torch.randn_like(z)
#                     z2 = torch.randn_like(z)
#                     dagp_loss_val = dagp_loss(self.generators[target_label], z1, z2, self.gamma)
                    
#                     # Total loss with adjusted weights
#                     g_loss = (
#                         -score_generated +  # Adversarial loss
#                         loss_label +       # Classification loss
#                         self.lst_weight * lst_loss +   # LST loss with reduced weight
#                         self.dagp_weight * dagp_loss_val  # DAGP loss with increased weight
#                     )
                    
#                     g_loss.backward()
#                     g_optimizers[target_label].step()
#                     epoch_g_loss += g_loss.item()
            
#             # Update anchors every epoch
#             self.update_anchors(dataset)
            
#             # Log metrics
#             if batch_count > 0:
#                 self.tracker.log_epoch(
#                     epoch=e+1,
#                     d_loss=epoch_d_loss/batch_count,
#                     g_loss=epoch_g_loss/batch_count,
#                     c_loss=epoch_c_loss/batch_count,
#                     lst_loss=lst_loss.item() if 'lst_loss' in locals() else 0.0,
#                     dagp_loss=dagp_loss_val.item() if 'dagp_loss_val' in locals() else 0.0
#                 )
            
#             # Visualize samples periodically
#             if e+1 in [100, 200, 300, 400, 500]:
#                 self.visualize_generated_samples(e)
        
#         print('')
#         self.cd.eval()
#         for i in self.generators:
#             i.eval()
        
#         # Plot metrics
#         self.tracker.plot_losses()
        
#         # Save models
#         save_dir = config.path_config.data / "trained_models"
#         save_dir.mkdir(parents=True, exist_ok=True)
        
#         torch.save(self.cd.state_dict(), save_dir / "cd_model.pth")
#         for i, generator in enumerate(self.generators):
#             torch.save(generator.state_dict(), save_dir / f"generator_{i}.pth")
        
#         print(f"\nModels saved to {save_dir}")
    
#     def generate_qualified_samples(self, target_label: int, num: int) -> torch.Tensor:
#         """Generate samples with diversity and feature preservation guarantees"""
#         self.generators[target_label].eval()
#         with torch.no_grad():
#             samples = []
#             while len(samples) < num:
#                 z = torch.randn(config.GAN_config.batch_size, config.GAN_config.z_size, device=config.device)
#                 gen_samples = self.generators[target_label].generate_samples(config.GAN_config.batch_size)
                
#                 # Apply diversity check
#                 if len(samples) > 0:
#                     last_samples = torch.stack(samples[-config.GAN_config.batch_size:])
#                     dists = torch.abs(gen_samples.unsqueeze(1) - last_samples.unsqueeze(0)).sum(dim=2)
#                     valid_mask = (dists > self.gamma).all(dim=1)
#                     gen_samples = gen_samples[valid_mask]
                
#                 samples.extend(gen_samples.cpu().numpy())
            
#             return torch.tensor(samples[:num], device=config.device)

#     def visualize_generated_samples(self, epoch):
#         """Visualize generated samples using t-SNE"""
#         with torch.no_grad():
#             # Create visualizer instance
#             visualizer = Visualizer(self)
#             # Generate visualization
#             visualizer.visualize_and_save(
#                 method='tsne',
#                 num_per_class=100,
#                 out_dir=config.path_config.GAN_out
#             )
            
#             # Set models back to training mode
#             for i in self.generators:
#                 i.train() 