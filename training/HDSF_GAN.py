# import random
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.nn.functional import cross_entropy, cosine_similarity
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# import pandas as pd
# from typing import Dict, List, Tuple
# import os
# from sklearn.manifold import TSNE

# from training import config, datasets, models
# from scripts.visualize_gan_output import Visualizer
# from training.tracker import TrainingTracker

# class ProgressiveAnchorRefinement:
#     def __init__(self, feature_dim: int, num_classes: int, momentum: float = 0.9):
#         self.feature_dim = feature_dim
#         self.num_classes = num_classes
#         self.momentum = momentum
#         self.anchors = torch.zeros(num_classes, feature_dim).to(config.device)
#         self.initialized = [False] * num_classes
        
#     def update_anchors(self, features: torch.Tensor, labels: torch.Tensor):
#         """Update class anchors using moving average."""
#         for c in range(self.num_classes):
#             mask = (labels == c)
#             if mask.sum() > 0:
#                 class_features = features[mask]
#                 if not self.initialized[c]:
#                     self.anchors[c] = class_features.mean(dim=0)
#                     self.initialized[c] = True
#                 else:
#                     self.anchors[c] = (self.momentum * self.anchors[c] + 
#                                      (1 - self.momentum) * class_features.mean(dim=0))

# class AdaptiveLossScheduler:
#     def __init__(self, initial_weights: Dict[str, float], 
#                  min_weight: float = 0.1, max_weight: float = 10.0):
#         self.weights = initial_weights
#         self.min_weight = min_weight
#         self.max_weight = max_weight
#         self.history = {k: [] for k in initial_weights.keys()}
        
#     def update_weights(self, metrics: Dict[str, float]):
#         """Update loss weights based on training metrics."""
#         for k, v in metrics.items():
#             if k in self.weights:
#                 self.history[k].append(v)
#                 if len(self.history[k]) > 1:
#                     trend = self.history[k][-1] - self.history[k][-2]
#                     self.weights[k] = np.clip(
#                         self.weights[k] * (1 + 0.1 * np.sign(trend)),
#                         self.min_weight,
#                         self.max_weight
#                     )

# class HDSFGAN:
#     def __init__(self):
#         # Initialize multiple generators (one per class)
#         self.generators = [
#             models.GeneratorModel(config.GAN_config.z_size, datasets.feature_num).to(config.device)
#             for _ in range(datasets.label_num)
#         ]
        
#         # Initialize CD model with feature extractor
#         self.cd = models.CDModel(datasets.feature_num, datasets.label_num).to(config.device)
        
#         # Initialize Progressive Anchor Refinement
#         self.par = ProgressiveAnchorRefinement(
#             feature_dim=self.cd.output_dim,
#             num_classes=datasets.label_num
#         )
        
#         # Initialize Adaptive Loss Scheduler
#         self.loss_scheduler = AdaptiveLossScheduler({
#             'lst': 1.0,      # Latent Space Triangulation
#             'intra': 0.5,    # Intra-class similarity
#             'inter': 0.5,    # Inter-class separation
#             'sgc': 0.1       # Stability-aware Gradient Control
#         })
        
#         self.samples = dict()
#         self.tracker = TrainingTracker()
        
#     def compute_lst_loss(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
#         """Compute Latent Space Triangulation Loss."""
#         loss = 0.0
#         for c in range(datasets.label_num):
#             mask = (labels == c)
#             if mask.sum() > 0:
#                 class_features = features[mask]
#                 distances = torch.norm(class_features - self.par.anchors[c], dim=1)
#                 loss += distances.mean()
#         return loss / datasets.label_num
    
#     def compute_cosine_losses(self, features: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         """Compute enhanced cosine similarity losses."""
#         # Normalize features
#         features_norm = F.normalize(features, dim=1)
        
#         # Intra-class similarity
#         intra_loss = torch.tensor(0.0, device=config.device)
#         for c in range(datasets.label_num):
#             mask = (labels == c)
#             if mask.sum() > 0:
#                 class_features = features_norm[mask]
#                 # Compute pairwise similarities within class
#                 sim_matrix = torch.mm(class_features, class_features.t())
#                 # Remove self-similarities
#                 sim_matrix.fill_diagonal_(0)
#                 # Use a margin-based loss to encourage higher similarity
#                 # We want similarities to be close to 1.0 for same class
#                 intra_loss = intra_loss + F.relu(0.8 - sim_matrix).mean()
        
#         # Inter-class separation
#         inter_loss = torch.tensor(0.0, device=config.device)
#         for c1 in range(datasets.label_num):
#             for c2 in range(c1 + 1, datasets.label_num):
#                 mask1 = (labels == c1)
#                 mask2 = (labels == c2)
#                 if mask1.sum() > 0 and mask2.sum() > 0:
#                     features1 = features_norm[mask1]
#                     features2 = features_norm[mask2]
#                     # Compute similarities between classes
#                     sim_matrix = torch.mm(features1, features2.t())
#                     # Use a larger margin for inter-class separation
#                     # We want similarities to be close to 0.0 for different classes
#                     margin = 0.3  # Increased from 0.5 to make separation more strict
#                     inter_loss = inter_loss + F.relu(sim_matrix - margin).mean()
        
#         # Scale the losses to make them more comparable
#         intra_loss = intra_loss / datasets.label_num
#         inter_loss = inter_loss / (datasets.label_num * (datasets.label_num - 1) / 2)
        
#         # Add L2 regularization to prevent feature collapse
#         l2_reg = 0.01 * torch.norm(features, p=2)
        
#         return intra_loss + l2_reg, inter_loss
    
#     def compute_sgc_loss(self, generator: nn.Module, batch_size: int) -> torch.Tensor:
#         """Compute Stability-aware Gradient Control Loss."""
#         # Generate two sets of samples with different random noise
#         z1 = torch.randn(batch_size, config.GAN_config.z_size, requires_grad=True).to(config.device)
#         z2 = torch.randn(batch_size, config.GAN_config.z_size, requires_grad=True).to(config.device)
        
#         # Generate samples
#         samples1 = generator(z1)
#         samples2 = generator(z2)
        
#         # Compute feature differences
#         features1 = self.cd.main_model(samples1)
#         features2 = self.cd.main_model(samples2)
        
#         # Compute stability loss
#         stability_loss = F.relu(0.1 - torch.norm(features1 - features2, dim=1)).mean()
        
#         # Add Jacobian regularization
#         # Compute sum of outputs for gradient computation
#         sum_outputs = samples1.sum()
#         # Compute gradients with respect to input
#         jacobian = torch.autograd.grad(sum_outputs, z1, create_graph=True)[0]
#         jacobian_reg = torch.norm(jacobian, dim=1).mean()
        
#         return stability_loss + 0.1 * jacobian_reg
    
#     def fit(self, dataset):
#         """Training loop for HDSF-GAN."""
#         self.cd.train()
#         for g in self.generators:
#             g.train()
        
#         self.divideSamples(dataset)
        
#         # Initialize optimizers
#         cd_optimizer = torch.optim.Adam(
#             params=self.cd.parameters(),
#             lr=config.GAN_config.cd_lr,
#             betas=(0.5, 0.999)
#         )
        
#         g_optimizers = [
#             torch.optim.Adam(
#                 params=g.parameters(),
#                 lr=config.GAN_config.g_lr,
#                 betas=(0.5, 0.999)
#             )
#             for g in self.generators
#         ]
        
#         # Get total epochs from config
#         total_epochs = config.GAN_config.epochs
        
#         try:
#             for epoch in range(total_epochs):
#                 if epoch >= total_epochs:
#                     break
                    
#                 print(f'\r{(epoch + 1) / total_epochs: .2%}', end='')
                
#                 epoch_metrics = {
#                     'd_loss': torch.tensor(0.0, device=config.device),
#                     'g_loss': torch.tensor(0.0, device=config.device),
#                     'c_loss': torch.tensor(0.0, device=config.device),
#                     'lst_loss': torch.tensor(0.0, device=config.device),
#                     'intra_loss': torch.tensor(0.0, device=config.device),
#                     'inter_loss': torch.tensor(0.0, device=config.device),
#                     'sgc_loss': torch.tensor(0.0, device=config.device)
#                 }
#                 batch_count = 0
                
#                 for target_label in self.samples.keys():
#                     # Train CD model
#                     for _ in range(config.GAN_config.cd_loopNo):
#                         cd_optimizer.zero_grad()
                        
#                         # Get real and generated samples
#                         real_samples = self.get_target_samples(target_label, config.GAN_config.batch_size)
#                         generated_samples = self.generators[target_label].generate_samples(config.GAN_config.batch_size)
                        
#                         # Forward pass
#                         score_real, predicted_labels = self.cd(real_samples)
#                         score_generated = self.cd(generated_samples)[0]
                        
#                         # Compute losses
#                         d_loss = (score_generated - score_real).mean() / 2
#                         c_loss = cross_entropy(
#                             predicted_labels,
#                             torch.full([len(predicted_labels)], target_label, device=config.device),
#                             weight=datasets.class_weights.to(config.device)
#                         )
                        
#                         # Update CD model
#                         loss = d_loss + c_loss
#                         loss.backward()
#                         cd_optimizer.step()
                        
#                         epoch_metrics['d_loss'] += d_loss.detach()
#                         epoch_metrics['c_loss'] += c_loss.detach()
#                         batch_count += 1
                    
#                     # Train Generator
#                     for _ in range(config.GAN_config.g_loopNo):
#                         g_optimizers[target_label].zero_grad()
                        
#                         # Generate samples
#                         generated_samples = self.generators[target_label].generate_samples(config.GAN_config.batch_size)
#                         real_samples = self.get_target_samples(target_label, config.GAN_config.batch_size)
                        
#                         # Get features
#                         features_real = self.cd.main_model(real_samples)
#                         features_gen = self.cd.main_model(generated_samples)
                        
#                         # Update anchors
#                         self.par.update_anchors(features_real.detach(), 
#                                               torch.full([len(real_samples)], target_label, device=config.device))
                        
#                         # Compute all losses
#                         score_generated = self.cd(generated_samples)[0].mean()
#                         lst_loss = self.compute_lst_loss(features_gen, 
#                                                        torch.full([len(generated_samples)], target_label, device=config.device))
#                         intra_loss, inter_loss = self.compute_cosine_losses(features_gen, 
#                                                                            torch.full([len(generated_samples)], target_label, device=config.device))
#                         sgc_loss = self.compute_sgc_loss(self.generators[target_label], config.GAN_config.batch_size)
                        
#                         # Combine losses with adaptive weights
#                         g_loss = (-score_generated + 
#                                  self.loss_scheduler.weights['lst'] * lst_loss +
#                                  self.loss_scheduler.weights['intra'] * intra_loss +
#                                  self.loss_scheduler.weights['inter'] * inter_loss +
#                                  self.loss_scheduler.weights['sgc'] * sgc_loss
#                                 )
                        
#                         g_loss.backward(retain_graph=True)
#                         g_optimizers[target_label].step()
                        
#                         # Store detached losses for metrics
#                         epoch_metrics['g_loss'] += g_loss.detach()
#                         epoch_metrics['lst_loss'] += lst_loss.detach()
#                         epoch_metrics['intra_loss'] += intra_loss.detach()
#                         epoch_metrics['inter_loss'] += inter_loss.detach()
#                         epoch_metrics['sgc_loss'] += sgc_loss.detach()
                
#                 # Update loss weights
#                 if batch_count > 0:
#                     metrics = {k: v.item()/batch_count for k, v in epoch_metrics.items()}
#                     self.loss_scheduler.update_weights(metrics)
                    
#                     # Log metrics
#                     self.tracker.log_epoch(
#                         epoch=epoch+1,
#                         **metrics
#                     )
                
#                 # Visualize samples periodically
#                 if epoch + 1 in [250, 500, 750, 1000, 1250, 1500, 1750, 2000]:
#                     self.visualize_generated_samples(epoch + 1)
            
#             print('\nTraining completed!')
            
#         except KeyboardInterrupt:
#             print('\nTraining interrupted by user!')
            
#         finally:
#             # Set models to eval mode
#             self.cd.eval()
#             for g in self.generators:
#                 g.eval()
            
#             # Create output directory
#             output_dir = config.path_config.GAN_out
#             os.makedirs(output_dir, exist_ok=True)
            
#             # Plot and save all metrics
#             print("\nSaving training metrics and plots...")
#             self.tracker.plot_metrics(output_dir)
            
#             # Save models
#             save_dir = config.path_config.data / "trained_models"
#             save_dir.mkdir(parents=True, exist_ok=True)
            
#             torch.save(self.cd.state_dict(), save_dir / "hdsf_cd_model.pth")
#             for i, generator in enumerate(self.generators):
#                 torch.save(generator.state_dict(), save_dir / f"hdsf_generator_{i}.pth")
            
#             print(f"\nModels and training plots saved to {save_dir}")
#             print(f"Training metrics and plots saved to {output_dir}")
    
#     def divideSamples(self, dataset: datasets.TrDataset) -> None:
#         """Divide dataset into class-specific samples."""
#         for sample, label in dataset:
#             label = label.item()
#             if label not in self.samples.keys():
#                 self.samples[label] = sample.unsqueeze(0)
#             else:
#                 self.samples[label] = torch.cat([self.samples[label], sample.unsqueeze(0)])
    
#     def get_target_samples(self, label: int, num: int) -> torch.Tensor:
#         """Get random samples for a specific class."""
#         return torch.stack(
#             random.choices(
#                 self.samples[label],
#                 k=num
#             )
#         )
    
#     def generate_samples(self, target_label: int, num: int) -> torch.Tensor:
#         """Generate samples for a specific class."""
#         return self.generators[target_label].generate_samples(num)
    
#     def visualize_generated_samples(self, epoch: int):
#         """Visualize generated samples for all classes."""
#         with torch.no_grad():
#             real_data = []
#             real_labels = []
#             fake_data = []
#             fake_labels = []

#             for label in range(datasets.label_num):
#                 # Get real samples
#                 real_samples = self.get_target_samples(label, 100)
#                 real_data.append(real_samples.cpu().numpy())
#                 real_labels.extend([label] * 100)

#                 # Get generated samples
#                 gen_samples = self.generate_samples(label, 100)
#                 fake_data.append(gen_samples.cpu().numpy())
#                 fake_labels.extend([label] * 100)

#             real_data = np.concatenate(real_data, axis=0)
#             fake_data = np.concatenate(fake_data, axis=0)
#             all_data = np.concatenate([real_data, fake_data], axis=0)
#             all_labels = np.array(real_labels + fake_labels)
#             domain_labels = np.array(['Real'] * len(real_data) + ['Fake'] * len(fake_data))

#             # t-SNE visualization
#             tsne = TSNE(n_components=2, perplexity=30, random_state=42)
#             embeddings = tsne.fit_transform(all_data)

#             # Real vs Fake visualization
#             plt.figure(figsize=(10, 8))
#             for domain in ['Real', 'Fake']:
#                 idx = domain_labels == domain
#                 plt.scatter(embeddings[idx, 0], embeddings[idx, 1], label=domain, alpha=0.6, s=20)
#             plt.title(f't-SNE Visualization of Real vs Generated Samples (Epoch {epoch})')
#             plt.legend()
#             plt.grid()
#             plt.savefig(os.path.join('data/GAN_output', f'tsne_real_vs_fake_epoch_{epoch}.png'))
#             plt.close()

#             # Per Class visualization
#             plt.figure(figsize=(10, 8))
#             for label in range(datasets.label_num):
#                 idx = all_labels == label
#                 plt.scatter(embeddings[idx, 0], embeddings[idx, 1], label=f'Class {label}', alpha=0.6, s=20)
#             plt.title(f't-SNE Per Class Visualization (Epoch {epoch})')
#             plt.legend()
#             plt.grid()
#             plt.savefig(os.path.join('data/GAN_output', f'tsne_per_class_epoch_{epoch}.png'))
#             plt.close()

#             # Set models back to training mode
#             for g in self.generators:
#                 g.train() 