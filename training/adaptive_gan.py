# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import random
# from training.utils import init_weights, one_hot
# from training import config, datasets
# from training.models.t_gan_g import Generator
# from training.models.cd_model import CDModel
# from training.tracker import TrainingTracker
# from scripts.visualize_gan_output import Visualizer

# class AdaptiveGAN:
#     def __init__(self, z_size: int, class_num: int, feature_num: int):
#         self.generator = Generator(z_size, class_num, feature_num).to(config.device)
#         self.cd = CDModel(feature_num, class_num).to(config.device)
        
#         # Initialize class centers
#         self.class_centers = torch.zeros(class_num, 16).to(config.device)  # 16 is the feature dimension from CDModel
#         self.center_momentum = config.GAN_config.center_momentum  # Momentum for updating centers
        
#         # Initialize class weights
#         self.class_weights = torch.ones(class_num).to(config.device)
#         self.alpha = config.GAN_config.alpha  # Balancing aggressiveness parameter
        
#         # Store samples for each class
#         self.samples = {}
        
#         self.tracker = TrainingTracker("Adaptive_GAN_Training")
#         self.visualizer = Visualizer(self)

#     def update_class_weights(self, class_counts):
#         """Update class weights based on inverse frequency"""
#         for label, count in class_counts.items():
#             self.class_weights[label] = 1.0 / (count ** self.alpha)
#         # Normalize weights
#         self.class_weights = self.class_weights / self.class_weights.mean()

#     def update_class_centers(self, features, labels):
#         """Update class centers using momentum"""
#         with torch.no_grad():
#             for label in range(self.class_centers.size(0)):
#                 mask = (labels == label)
#                 if mask.sum() > 0:
#                     center = features[mask].mean(dim=0)
#                     self.class_centers[label] = (
#                         self.center_momentum * self.class_centers[label] +
#                         (1 - self.center_momentum) * center
#                     )
#                     # Normalize updated center
#                     self.class_centers[label] = F.normalize(self.class_centers[label], p=2, dim=0)

#     def compute_separation_loss(self):
#         """Compute distance-based separation loss between class centers, normalized between 0 and 1"""
#         n_classes = self.class_centers.size(0)
#         total_term_sum = 0.0
#         count = 0

#         if n_classes < 2:
#             return 0.0 # No pairs to compute separation for

#         for i in range(n_classes):
#             for j in range(i + 1, n_classes):
#                 # Compute Euclidean distance between centers
#                 # Since class_centers are normalized to unit sphere, max distance is 2
#                 distance = torch.norm(self.class_centers[i] - self.class_centers[j], p=2)
                
#                 # Normalize each term: 1 when distance is 0, 0 when distance is 2
#                 # Clamping distance to ensure it's within [0, 2] due to potential floating point inaccuracies
#                 term = 1.0 - (torch.clamp(distance, 0.0, 2.0) / 2.0)
#                 total_term_sum += term
#                 count += 1
        
#         # Normalize the sum by the number of pairs to get a value between 0 and 1
#         return total_term_sum / count if count > 0 else 0.0

#     def divideSamples(self, dataset):
#         """Divide samples by class and initialize class weights"""
#         class_counts = {}
#         for sample, label in dataset:
#             label = label.item()
#             if label not in self.samples:
#                 self.samples[label] = sample.unsqueeze(0)
#                 class_counts[label] = 1
#             else:
#                 self.samples[label] = torch.cat([self.samples[label], sample.unsqueeze(0)])
#                 class_counts[label] = class_counts.get(label, 0) + 1
        
#         # Update initial class weights
#         self.update_class_weights(class_counts)

#     def fit(self, dataset):
#         self.generator.train()
#         self.cd.train()

#         # Divide samples and initialize weights
#         self.divideSamples(dataset)

#         cd_optimizer = torch.optim.Adam(self.cd.parameters(), lr=config.GAN_config.cd_lr, betas=(0.5, 0.999))
#         g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=config.GAN_config.g_lr, betas=(0.5, 0.999))

#         for epoch in range(config.GAN_config.epochs):
#             print(f'\rEpoch [{epoch + 1}/{config.GAN_config.epochs}]', end='')
            
#             epoch_d_loss, epoch_g_loss, epoch_c_loss, epoch_sep_loss = 0.0, 0.0, 0.0, 0.0
#             total_steps = 0

#             for label in self.samples.keys():
#                 batch_size = config.GAN_config.batch_size
#                 total_steps += 1

#                 # === Train Discriminator ===
#                 real_data = self.get_target_samples(label, batch_size).to(config.device)
#                 real_labels = torch.full((batch_size,), label, dtype=torch.long).to(config.device)
#                 one_hot_labels = one_hot(real_labels, len(self.samples)).to(config.device)

#                 z = torch.randn(batch_size, config.GAN_config.z_size).to(config.device)
#                 gen_samples = self.generator(z, one_hot_labels)

#                 d_real, c_real = self.cd(real_data)
#                 d_fake, c_fake = self.cd(gen_samples.detach())

#                 # Wasserstein GAN loss
#                 d_loss = -(torch.mean(d_real) - torch.mean(d_fake))
#                 c_loss_real = F.cross_entropy(c_real, real_labels, weight=self.class_weights)

#                 cd_total_loss = d_loss + config.GAN_config.lambda_c * c_loss_real
#                 cd_optimizer.zero_grad()
#                 cd_total_loss.backward()
#                 cd_optimizer.step()

#                 # === Train Generator ===
#                 z = torch.randn(batch_size, config.GAN_config.z_size).to(config.device)
#                 gen_samples = self.generator(z, one_hot_labels)
#                 d_fake, c_fake = self.cd(gen_samples)

#                 g_adv_loss = -torch.mean(d_fake)
#                 c_fake_loss = F.cross_entropy(c_fake, real_labels, weight=self.class_weights)

#                 # Update class centers
#                 features = self.cd.main_model(gen_samples)
#                 self.update_class_centers(features, real_labels)

#                 # Compute separation loss
#                 sep_loss = self.compute_separation_loss()

#                 # Total generator loss with scaled separation loss
#                 g_total_loss = g_adv_loss + config.GAN_config.lambda_c * c_fake_loss + config.GAN_config.lambda_sep * sep_loss

#                 g_optimizer.zero_grad()
#                 g_total_loss.backward()
#                 g_optimizer.step()

#                 # Update metrics
#                 epoch_d_loss += d_loss.item()
#                 epoch_g_loss += g_total_loss.item()
#                 epoch_c_loss += c_fake_loss.item()
#                 epoch_sep_loss += sep_loss.item()

#             # Average metrics
#             epoch_d_loss /= total_steps
#             epoch_g_loss /= total_steps
#             epoch_c_loss /= total_steps
#             epoch_sep_loss /= total_steps

#             # Log metrics
#             self.tracker.log_epoch(epoch+1, epoch_d_loss, epoch_g_loss, epoch_c_loss, epoch_sep_loss)

#             # Save visualizations at intervals
#             if (epoch + 1) in [100, 200, 300, 400, 500]:
#                 print(f"\nSaving metrics and visualizing after epoch {epoch + 1}")
#                 self.tracker.plot_interval_metrics(epoch + 1)
#                 self.visualizer.visualize_and_save(method='tsne', num_per_class=300)

#     def get_target_samples(self, label: int, num: int) -> torch.Tensor:
#         """Get real samples for a specific class"""
#         return torch.stack(
#             random.choices(self.samples[label], k=num)
#         )

#     def generate_samples(self, target_label: int, num: int):
#         """Generate samples for a specific class"""
#         z = torch.randn(num, config.GAN_config.z_size).to(config.device)
#         labels = torch.full((num,), target_label, dtype=torch.long).to(config.device)
#         one_hot_labels = one_hot(labels, len(self.samples)).to(config.device)
#         samples = self.generator(z, one_hot_labels)
#         return samples.cpu().detach() 