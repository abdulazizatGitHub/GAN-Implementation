import random
import torch
from torch.nn.functional import cross_entropy
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import math

from training import config, datasets, models
from scripts.visualize_gan_output import plot_gan_output_grid
from training.tracker import TrainingTracker

class TMGGANDPL:
    def __init__(self):
        self.generators = [
            models.GeneratorModel(config.GAN_config.z_size, datasets.feature_num).to(config.device)
            for _ in range(datasets.label_num)
        ]
        self.cd = models.CDModel(datasets.feature_num, datasets.label_num).to(config.device)
        self.samples = dict()
        self.tracker = TrainingTracker()
        # Compute imbalance ratios and dynamic weights
        class_counts = torch.bincount(torch.tensor([label for _, label in datasets.TrDataset()]))
        max_count = class_counts.max()
        self.ir_ratios = {i: (max_count / count).item() for i, count in enumerate(class_counts)}
        self.lambda_base = 0.1
        self.class_weights = {
            label: self.lambda_base * math.log(1 + ir)
            for label, ir in self.ir_ratios.items()
        }
    
    def pearson_correlation(self, f1, f2):
        f1_centered = f1 - f1.mean(dim=0, keepdim=True)
        f2_centered = f2 - f2.mean(dim=0, keepdim=True)
        cov = (f1_centered * f2_centered).mean()  # Direct mean over all elements
        stds = torch.sqrt(torch.sum(f1_centered**2)) * torch.sqrt(torch.sum(f2_centered**2))
        return cov / (stds + 1e-8)

    def calculate_dpl(self, target_label=None, batch_size=None, training=False):
        """Unified DPL calculation used for both training and logging"""
        if target_label is None:  # Logging case - calculate for all classes
            dpl_loss = 0
            inter_vals = []
            intra_vals = []
            
            for label in range(datasets.label_num):
                # Get samples
                gen_samples = self.generate_samples(label, batch_size or config.GAN_config.batch_size).to(config.device)
                real_samples = self.get_target_samples(label, batch_size or config.GAN_config.batch_size).to(config.device)
                
                # Get features
                F_gen = self.cd.main_model(gen_samples)
                F_real = self.cd.main_model(real_samples)
                
                # Intra-class (maximize)
                intra_sim = torch.sigmoid(self.pearson_correlation(F_gen, F_real))
                intra_vals.append(intra_sim.item())
                
                # Inter-class (minimize)
                other_gen_samples = torch.cat([
                    self.generate_samples(j, batch_size or config.GAN_config.batch_size).to(config.device)
                    for j in range(datasets.label_num) if j != label
                ])
                F_other_gen = self.cd.main_model(other_gen_samples)
                F_gen_rep = F_gen.repeat(other_gen_samples.shape[0] // F_gen.shape[0], 1)
                inter_sim = torch.sigmoid(self.pearson_correlation(F_gen_rep, F_other_gen))
                inter_vals.append(inter_sim.item())
                
                if training:
                    lambda_k = self.class_weights[label]
                    dpl_loss += lambda_k * (inter_sim - intra_sim)
            
            if not training:
                # For logging, return weighted averages
                weights = torch.tensor([self.class_weights[k] for k in range(datasets.label_num)])
                weights = weights / weights.sum()
                return (
                    np.average(inter_vals, weights=weights.numpy()),
                    np.average(intra_vals, weights=weights.numpy())
                )
            return dpl_loss
        
        else:  # Training case - calculate for specific target_label
            # Get samples
            gen_samples = self.generators[target_label].generate_samples(batch_size or config.GAN_config.batch_size)
            real_samples = self.get_target_samples(target_label, batch_size or config.GAN_config.batch_size)
            
            # Get features
            F_gen = self.cd.main_model(gen_samples)
            F_real = self.cd.main_model(real_samples)
            
            # Intra-class (maximize)
            intra_sim = torch.sigmoid(self.pearson_correlation(F_gen, F_real))
            
            # Inter-class (minimize)
            other_gen_samples = torch.cat([
                self.generators[j].generate_samples(batch_size or config.GAN_config.batch_size).to(config.device)
                for j in range(datasets.label_num) if j != target_label
            ])
            F_other_gen = self.cd.main_model(other_gen_samples)
            F_gen_rep = F_gen.repeat(other_gen_samples.shape[0] // F_gen.shape[0], 1)
            inter_sim = torch.sigmoid(self.pearson_correlation(F_gen_rep, F_other_gen))
            
            lambda_k = self.class_weights[target_label]
            return lambda_k * (inter_sim - intra_sim)

    def fit(self, dataset):
        self.cd.train()
        for i in self.generators:
            i.train()
        self.divideSamples(dataset)
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
            inter_pearson_vals = []
            intra_pearson_vals = []
            batch_count = 0
            for target_label in self.samples.keys():
                # Train C and D
                for _ in range(config.GAN_config.cd_loopNo):
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
                # Train G
                for _ in range(config.GAN_config.g_loopNo):
                    g_optimizers[target_label].zero_grad()
                    
                    # Generate samples and get scores
                    generated_samples = self.generators[target_label].generate_samples(config.GAN_config.batch_size)
                    real_samples = self.get_target_samples(target_label, config.GAN_config.batch_size)
                    
                    # Get CD outputs
                    self.cd(real_samples)
                    hidden_real = self.cd.hidden_status
                    score_generated, predicted_labels = self.cd(generated_samples)
                    hidden_generated = self.cd.hidden_status
                    
                    # Calculate losses
                    cd_hidden_loss = 0
                    if e >= 1000:
                        cd_hidden_loss = -self.pearson_correlation(hidden_real, hidden_generated)
                    
                    score_generated = score_generated.mean()
                    
                    # Classification loss
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
                    
                    # DPL loss (using unified calculation)
                    dpl_loss = self.calculate_dpl(target_label=target_label, training=True)
                    
                    # Combined loss
                    g_loss = -score_generated + cd_hidden_loss + dpl_loss
                    g_loss.backward()
                    g_optimizers[target_label].step()
                    epoch_g_loss += g_loss.item()
            
            # Logging (using same DPL calculation)
            inter_pearson, intra_pearson = self.calculate_dpl(batch_size=32)
            
            # Track metrics
            self.tracker.log_epoch(
                epoch=e+1,
                d_loss=epoch_d_loss/batch_count,
                g_loss=epoch_g_loss/batch_count,
                c_loss=epoch_c_loss/batch_count,
                inter_cos=inter_pearson,
                intra_cos=intra_pearson
            )
            if e + 1 in [50, 100, 150, 200]:
                self.visualize_generated_samples(e + 1)
        print('')
        self.cd.eval()
        for i in self.generators:
            i.eval()
        self.tracker.plot_metrics(config.path_config.dpl_tmg_gan_out)
        # --- Save trained models ---
        save_dir = config.path_config.dpl_tmg_gan_out / "trained_models"
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.cd.state_dict(), save_dir / "cd_model.pth")
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

    def visualize_generated_samples(self, epoch):
        with torch.no_grad():
            real_samples = []
            generated_samples = []
            for i in range(datasets.label_num):
                real = self.get_target_samples(i, 100).cpu().numpy()
                gen = self.generate_samples(i, 100).cpu().numpy()
                real_samples.append(real)
                generated_samples.append(gen)
            plot_gan_output_grid(
                real_samples, generated_samples, [np.empty((0, real_samples[0].shape[1])) for _ in range(datasets.label_num)], epoch,
                config.path_config.dpl_tmg_gan_out, n_classes=datasets.label_num, n_points=100
            )
            for i in self.generators:
                i.train() 