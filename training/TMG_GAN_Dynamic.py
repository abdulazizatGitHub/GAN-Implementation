import random
import torch
from torch.nn.functional import cross_entropy, cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import math

from training import config, datasets, models
from scripts.visualize_gan_output import plot_gan_output_grid
from training.tracker import TrainingTracker

class TMGGANDynamic:

    def __init__(self):
        self.generators = [
            models.GeneratorModel(config.GAN_config.z_size, datasets.feature_num).to(config.device)
            for _ in range (datasets.label_num)
        ]   

        self.cd = models.CDModel(datasets.feature_num, datasets.label_num).to(config.device)

        self.samples = dict()
        self.tracker = TrainingTracker()
        
        # Get imbalance ratios from dataset
        class_counts = torch.bincount(torch.tensor([label for _, label in datasets.TrDataset()]))
        max_count = class_counts.max()
        self.ir_ratios = {i: (max_count / count).item() for i, count in enumerate(class_counts)}
        self.lambda_base = 0.1  # Base weight for dynamic scaling

    def calculate_class_means(self):
        """Calculate mean features for each class."""
        class_means = {}
        with torch.no_grad():
            for label in range(datasets.label_num):
                if label in self.samples:
                    samples = self.samples[label]
                    features = self.cd.main_model(samples)
                    class_means[label] = features.mean(dim=0)
        return class_means

    def calculate_dynamic_weights(self):
        """Calculate dynamic weights based on dataset imbalance ratios."""
        self.class_weights = {
            label: self.lambda_base * math.log(1 + ir)
            for label, ir in self.ir_ratios.items()
        }
        return self.class_weights

    def fit(self, dataset, original_gan_samples=None):
        self.cd.train()

        for i in self.generators:
            i.train()
        
        self.divideSamples(dataset)
        self.calculate_dynamic_weights()  # Initialize dynamic weights

        cd_optimizer = torch.optim.Adam(
                params=self.cd.parameters(),
                lr=config.GAN_config.cd_lr,
                betas=(0.5, 0.999),
            )
        
        g_optimizers = [
            torch.optim.Adam(
                params=self.generators[i].parameters(),
                lr=config.GAN_config.g_lr,
                betas=(0.5, 0.999),
            )
            for i in range(datasets.label_num)
        ]

        for e in range(config.GAN_config.epochs):
            print(f'\r{(e + 1) / config.GAN_config.epochs: .2%}', end='')

            epoch_d_loss = 0.0
            epoch_g_loss = 0.0
            epoch_c_loss = 0.0
            inter_cosine_vals = []
            intra_cosine_vals = []
            batch_count = 0

            # Calculate class means for this epoch
            class_means = self.calculate_class_means()

            for target_label in self.samples.keys():
                #train C and D
                for _ in range (config.GAN_config.cd_loopNo):
                    cd_optimizer.zero_grad()
                    real_samples = self.get_target_samples(target_label, config.GAN_config.batch_size)
                    score_real, predicted_labels = self.cd(real_samples)
                    score_real = score_real.mean()
                    generated_samples = self.generators[target_label].generate_samples(config.GAN_config.batch_size)
                    score_generated = self.cd(generated_samples)[0].mean()
                    d_loss = (score_generated - score_real) / 2
                    if predicted_labels.shape[1] == len(datasets.class_weights):
                        c_loss = cross_entropy(
                            input=predicted_labels,
                            target=torch.full([len(predicted_labels)], target_label, device=config.device),
                            weight=datasets.class_weights.to(config.device)
                        )
                    else:
                        c_loss = cross_entropy(
                            input=predicted_labels,
                            target=torch.full([len(predicted_labels)], target_label, device=config.device)
                        )
                    loss = d_loss + c_loss
                    loss.backward()
                    cd_optimizer.step()
                    epoch_d_loss += d_loss.item()
                    epoch_c_loss += c_loss.item()
                    batch_count += 1
                
                #train G
                for _ in range(config.GAN_config.g_loopNo):
                    g_optimizers[target_label].zero_grad()
                    generated_samples = self.generators[target_label].generate_samples(config.GAN_config.batch_size)
                    real_samples = self.get_target_samples(target_label, config.GAN_config.batch_size)
                    self.cd(real_samples)
                    hidden_real = self.cd.hidden_status
                    score_generated, predicted_labels = self.cd(generated_samples)
                    hidden_generated = self.cd.hidden_status
                    cd_hidden_loss = - cosine_similarity(hidden_real, hidden_generated).mean()
                    score_generated = score_generated.mean()
                    if predicted_labels.shape[1] == len(datasets.class_weights):
                        loss_label = cross_entropy(
                            input=predicted_labels,
                            target=torch.full([len(predicted_labels)], target_label, device=config.device),
                            weight=datasets.class_weights.to(config.device)
                        )
                    else:
                        loss_label = cross_entropy(
                            input=predicted_labels,
                            target=torch.full([len(predicted_labels)], target_label, device=config.device)
                        )

                    if e < 1000:
                        cd_hidden_loss = 0
                    
                    # --- Dynamic Weighted Cosine Similarity Loss ---
                    # Calculate mean features for generated samples
                    gen_features = self.cd.main_model(generated_samples)
                    gen_mean = gen_features.mean(dim=0)
                    
                    # Calculate cosine loss with dynamic weights
                    cosine_loss = 0
                    for i in range(datasets.label_num):
                        if i != target_label:
                            # Compute inter-class similarity
                            cos_sim = cosine_similarity(gen_mean.unsqueeze(0), class_means[i].unsqueeze(0))
                            # Apply dynamic weight
                            weight = self.class_weights[target_label]
                            cosine_loss += weight * cos_sim
                    
                    # Subtract intra-class similarity (without extra weight)
                    intra_sim = cosine_similarity(gen_mean.unsqueeze(0), class_means[target_label].unsqueeze(0))
                    cosine_loss -= intra_sim

                    # --- Add to generator loss ---
                    g_loss = -score_generated + cd_hidden_loss + cosine_loss
                    g_loss.backward()
                    g_optimizers[target_label].step()
                    epoch_g_loss += g_loss.item()
            
            for i in g_optimizers:
                i.zero_grad()

            for i  in self.generators:
                i.generate_samples(3)
            
            g_hidden_losses = []

            for i, _ in enumerate(self.generators):
                for j, _ in enumerate(self.generators):
                    
                    if i == j:
                        continue
                    else:
                        g_hidden_losses.append(
                            cosine_similarity(
                                self.generators[i].hidden_status,
                                self.generators[j].hidden_status,
                            )
                        )
            
            g_hidden_losses = torch.stack(g_hidden_losses)
            inter_cosine = torch.mean(g_hidden_losses).item() if len(g_hidden_losses) > 0 else 0.0

            # Intra-class cosine similarity: pairwise between generated and real samples for each class
            intra_class_cosines = []
            with torch.no_grad():
                for k in range(datasets.label_num):
                    batch_size = min(len(self.samples[k]), 32)
                    gen_samples = self.generate_samples(k, batch_size).to(config.device)
                    real_samples = self.get_target_samples(k, batch_size).to(config.device)
                    F_gen = self.cd.main_model(gen_samples)
                    F_real = self.cd.main_model(real_samples)
                    # Normalize features
                    F_gen_norm = F_gen / F_gen.norm(dim=1, keepdim=True)
                    F_real_norm = F_real / F_real.norm(dim=1, keepdim=True)
                    # Pairwise cosine similarity (diagonal)
                    pairwise_cos = (F_gen_norm * F_real_norm).sum(dim=1).abs()
                    intra_class_cosines.append(pairwise_cos.mean().item())
            intra_cosine = float(np.mean(intra_class_cosines)) if len(intra_class_cosines) > 0 else 0.0
            g_hidden_losses.mean().backward()
            for i in g_optimizers:
                i.step()
            
            # Log and track metrics
            if batch_count > 0:
                self.tracker.log_epoch(
                    epoch=e+1,
                    d_loss=epoch_d_loss/batch_count,
                    g_loss=epoch_g_loss/batch_count,
                    c_loss=epoch_c_loss/batch_count,
                    inter_cos=inter_cosine,
                    intra_cos=intra_cosine
                )
            
            if e + 1 in [500, 1000, 1500, 2000]:
                self.visualize_generated_samples(e + 1, original_gan_samples=original_gan_samples)
        
        print('')
        self.cd.eval()

        for i in self.generators:
            i.eval()
        
        # Plot metrics at the end
        self.tracker.plot_metrics(config.path_config.dynamic_gan_out)

        # --- Save trained models ---
        save_dir = config.path_config.data / "trained_models"
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save CD model
        torch.save(self.cd.state_dict(), save_dir / "cd_model.pth")

        # Save Generator models
        for i, generator in enumerate(self.generators):
            torch.save(generator.state_dict(), save_dir / f"generator_{i}.pth")

        print(f"\nModels saved to {save_dir}")
    

    def divideSamples(self, dataset: datasets.TrDataset) -> None:
        for sample, label in dataset:
            label = label.item()
            if label not in self.samples.keys():
                self.samples[label] = sample.unsqueeze(0)
            else:
                self.samples[label] = torch.cat([self.samples[label], sample.unsqueeze(0)])
    

    def get_target_samples(self, label: int, num: int) -> torch.Tensor:
        return torch.stack(
            random.choices(
                self.samples[label],
                k=num,
            )
        )
    

    def generate_samples(self, target_label: int, num: int):
        return self.generators[target_label].generate_samples(num).cpu().detach()
    

    def generate_qualified_samples(self, target_label: int, num: int):
        result = []
        patience = 10

        while len(result) < num:
            sample = self.generators[target_label].generate_samples(1)
            label = torch.argmax(self.cd(sample)[1])

            if label == target_label or patience == 0:
                result.append(sample.cpu().detach())
                patience = 10
            else:
                patience -= 1
        
        return torch.cat(result)

    def visualize_generated_samples(self, epoch, original_gan_samples=None):
        with torch.no_grad():
            real_samples = []
            dynamic_gen_samples = [] # Renamed from generated_samples to clarify it's from Dynamic GAN
            
            # If original_gan_samples are not provided, initialize as empty lists of 2D arrays
            if original_gan_samples is None:
                # Need to determine the feature dimension D dynamically or assume a default.
                # For now, let's assume it has the same feature_num as the dataset features.
                original_gan_samples = [np.empty((0, datasets.feature_num)) for _ in range(datasets.label_num)]

            for i in range(datasets.label_num):
                real = self.get_target_samples(i, 100).cpu().numpy()
                dyn_gen = self.generate_samples(i, 100).cpu().numpy() # Samples from TMG_GAN_Dynamic
                real_samples.append(real)
                dynamic_gen_samples.append(dyn_gen)
            
            # Ensure original_gan_samples are also numpy arrays if they were provided as tensors
            original_gan_samples_np = []
            for i in range(datasets.label_num):
                if isinstance(original_gan_samples[i], torch.Tensor):
                    original_gan_samples_np.append(original_gan_samples[i].cpu().numpy())
                else:
                    original_gan_samples_np.append(original_gan_samples[i])

            plot_gan_output_grid(
                real_samples, original_gan_samples_np, dynamic_gen_samples, epoch,
                config.path_config.dynamic_gan_out, n_classes=datasets.label_num, n_points=100
            )
            
            for i in self.generators:
                i.train()
