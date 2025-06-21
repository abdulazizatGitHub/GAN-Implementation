# import random
# import torch
# from torch.nn.functional import cross_entropy, cosine_similarity

# from training import config, datasets, models
# from scripts.visualize_gan_output import Visualizer
# from training.tracker import TrainingTracker
# from training.utils import one_hot

# class TGAN:
#     def __init__(self):
#         self.generator = models.Generator(config.GAN_config.z_size, datasets.label_num, datasets.feature_num).to(config.device)
#         self.cd = models.CDModel(datasets.feature_num, datasets.label_num).to(config.device)

#         self.samples = dict()
#         self.time_weights = dict()
#         self.tracker = TrainingTracker()

#         self.visualizer = Visualizer(self)
    
#     def fit(self):
#         self.generator.train()
#         self.cd.train()

#         cd_optimizer = torch.optim.Adam(self.cd.parameters(), lr=config.GAN_config.cd_lr, betas=(0.5, 0.999))
#         g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=config.GAN_config.g_lr, betas=(0.5, 0.999))

#         total_time_weight = sum(self.time_weights.values())

#         for epoch in range(config.GAN_config.epochs):
#             print(f'\rEpoch [{epoch + 1}/{config.GAN_config.epochs}]', end='')

#             epoch_d_loss, epoch_g_loss, epoch_c_loss = 0.0, 0.0, 0.0
#             epoch_intra_sim, epoch_inter_sim = 0.0, 0.0
#             total_steps = 0

#             for label in self.samples.keys():
#                 tk = self.time_weights[label]
#                 steps = int(tk / total_time_weight * config.GAN_config.steps_per_epoch)

#                 for step in range(steps):
#                     batch_size = config.GAN_config.batch_size
#                     total_steps += 1

#                     # === Train Discriminator ===
#                     real_data = self.get_target_samples(label, batch_size).to(config.device)
#                     real_labels = torch.full((batch_size,), label, dtype=torch.long).to(config.device)

#                     z = torch.randn(batch_size, config.GAN_config.z_size).to(config.device)
#                     one_hot_labels = one_hot(real_labels, datasets.label_num).to(config.device)
#                     gen_samples = self.generator(z, one_hot_labels)

#                     d_real, c_real = self.cd(real_data)
#                     d_fake, c_fake = self.cd(gen_samples.detach())

#                     d_loss = -(torch.mean(d_real) - torch.mean(d_fake))  # WGAN loss
#                     c_loss_real = cross_entropy(c_real, real_labels)

#                     cd_total_loss = d_loss + config.GAN_config.lambda_c * c_loss_real
#                     cd_optimizer.zero_grad()
#                     cd_total_loss.backward()
#                     cd_optimizer.step()

#                     # === Train Generator ===
#                     z = torch.randn(batch_size, config.GAN_config.z_size).to(config.device)
#                     one_hot_labels = one_hot(real_labels, datasets.label_num).to(config.device)
#                     gen_samples = self.generator(z, one_hot_labels)

#                     d_fake, c_fake = self.cd(gen_samples)

#                     g_adv_loss = -torch.mean(d_fake)
#                     c_fake_loss = cross_entropy(c_fake, real_labels)

#                     # === Cosine Similarity Loss ===
#                     # Feature extraction
#                     F_real = self.cd.main_model(real_data)
#                     F_gen = self.cd.main_model(gen_samples)

#                     # Normalize (with numerical stability)
#                     epsilon = 1e-8
#                     F_real_norm = F_real / (F_real.norm(dim=1, keepdim=True) + epsilon)
#                     F_gen_norm = F_gen / (F_gen.norm(dim=1, keepdim=True) + epsilon)

#                     # Intra-class cosine similarity
#                     intra_cos_vals = (F_real_norm * F_gen_norm).sum(dim=1).abs()
#                     intra_sim = intra_cos_vals.mean()

#                     # Inter-class cosine similarity
#                     inter_cos_list = []
#                     num_classes = len(self.samples.keys())
                    
#                     for other_label in self.samples.keys():
#                         if other_label == label:
#                             continue

#                         # Generate samples for other class
#                         z_other = torch.randn(batch_size, config.GAN_config.z_size).to(config.device)
#                         one_hot_other = one_hot(
#                             torch.full((batch_size,), other_label, dtype=torch.long).to(config.device),
#                             datasets.label_num
#                         ).to(config.device)
#                         gen_other = self.generator(z_other, one_hot_other)
#                         F_gen_other = self.cd.main_model(gen_other)
#                         F_gen_other_norm = F_gen_other / (F_gen_other.norm(dim=1, keepdim=True) + epsilon)

#                         # Compute cosine similarities for each sample in the batch
#                         for i in range(batch_size):
#                             # Get the feature vector for current sample
#                             current_feat = F_gen_norm[i:i+1]  # Shape: [1, feature_dim]
                            
#                             # Compute cosine similarity with all samples from other class
#                             cos_sim = torch.mm(current_feat, F_gen_other_norm.t()).abs()  # Shape: [1, batch_size]
                            
#                             # Take mean over all similarities for this sample
#                             inter_cos_list.append(cos_sim.mean())

#                     if inter_cos_list:
#                         # Stack all similarities and take mean
#                         inter_sim = torch.stack(inter_cos_list).mean()
#                     else:
#                         inter_sim = torch.tensor(0.0, device=config.device)

#                     # Modified loss function with stronger inter-class separation
#                     margin = 0.3  # Increased margin for stronger separation
#                     cosine_loss = torch.relu(inter_sim - intra_sim + margin) + 0.1 * inter_sim  # Added direct penalty on inter_sim

#                     # Total generator loss
#                     g_total_loss = g_adv_loss + config.GAN_config.lambda_c * c_fake_loss + cosine_loss

#                     g_optimizer.zero_grad()
#                     g_total_loss.backward()
#                     g_optimizer.step()

#                     # Update epoch metrics
#                     epoch_d_loss += d_loss.item()
#                     epoch_c_loss += c_fake_loss.item()
#                     epoch_g_loss += g_total_loss.item()
#                     epoch_intra_sim += intra_sim.item()
#                     epoch_inter_sim += inter_sim.item()

#             # Average the metrics over all steps
#             epoch_d_loss /= total_steps
#             epoch_g_loss /= total_steps
#             epoch_c_loss /= total_steps
#             epoch_intra_sim /= total_steps
#             epoch_inter_sim /= total_steps

#             self.tracker.log_epoch(epoch+1, epoch_d_loss, epoch_g_loss, epoch_c_loss, epoch_intra_sim, epoch_inter_sim)

#             # Save plots at specified intervals
#             if (epoch + 1) in [10, 20, 30, 40, 50]:
#                 print(f"\nSaving metrics and visualizing after epoch {epoch + 1}")
#                 # Save interval-specific plots
#                 self.tracker.plot_interval_metrics(epoch + 1)
#                 # Generate and save t-SNE visualization
#                 self.visualizer.visualize_and_save(method='tsne', num_per_class=300)

#     def divideSamples(self, dataset: datasets.TrDataset) -> None:
#         for sample, label in dataset:
#             label = label.item()
#             if label not in self.samples.keys():
#                 self.samples[label] = sample.unsqueeze(0)
#             else:
#                 self.samples[label] = torch.cat([self.samples[label], sample.unsqueeze(0)])
        
#         # Define time weights manually for your imbalance:
#         custom_time_weights = {
#             0: 1.0,
#             1: 1.5,
#             2: 1.75,
#             3: 2.0,   # or 3.0 depending on your imbalance ratio
#             4: 3.0    # or 4.0 depending on your imbalance ratio
#         }

#         # Assign time weights
#         self.time_weights = {
#             label: custom_time_weights.get(label, 1.0)  # default 1.0 for safety
#             for label in self.samples.keys()
#         }


#     def get_target_samples(self, label: int, num: int) -> torch.Tensor:
#         return torch.stack(
#             random.choices(self.samples[label], k=num)
#         )
    
#     def generate_samples(self, target_label: int, num: int):
#         z = torch.randn(num, config.GAN_config.z_size).to(config.device)
#         labels = torch.full((num,), target_label, dtype=torch.long).to(config.device)
#         one_hot_labels = one_hot(labels, datasets.label_num).to(config.device)
#         samples = self.generator(z, one_hot_labels)
#         return samples.cpu().detach()

#     def generate_qualified_samples(self, target_label: int, num: int):
#         result = []
#         patience = 10

#         while len(result) < num:
#             z = torch.randn(1, config.GAN_config.z_size).to(config.device)
#             labels = torch.full((1,), target_label, dtype=torch.long).to(config.device)
#             one_hot_labels = one_hot(labels, datasets.label_num).to(config.device)
#             sample = self.generator(z, one_hot_labels)
#             _, class_logits = self.cd(sample)
#             predicted_label = torch.argmax(class_logits, dim=1).item()

#             if predicted_label == target_label or patience == 0:
#                 result.append(sample.cpu().detach())
#                 patience = 10
#             else:
#                 patience -= 1

#         return torch.cat(result, dim=0)
