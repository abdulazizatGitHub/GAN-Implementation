import random
import torch
from torch.nn.functional import cross_entropy, normalize, relu
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import math

from training import config, datasets, models
from scripts.visualize_gan_output import plot_gan_output_grid, plot_class_tsne
from training.tracker import TrainingTracker
from torch.optim.lr_scheduler import StepLR, MultiStepLR

class MYMETHOD:
    def __init__(self):
        self.generators = [
            models.GeneratorModel(config.GAN_config.z_size, datasets.feature_num).to(config.device)
            for _ in range(datasets.label_num)
        ]
        self.cd = models.CDModel(datasets.feature_num, datasets.label_num).to(config.device)
        self.samples = dict()
        self.tracker = TrainingTracker()
        # Loss weights
        self.lambda_adv = 1.0
        self.lambda_cls = 1.0
        self.lambda_fm = 70.0  # Feature matching loss weight

        # Calculate class imbalance ratios and dynamic class weights
        class_counts = torch.bincount(torch.tensor([label for _, label in datasets.TrDataset()]))
        max_count = class_counts.max()
        self.ir_ratios = {i: (max_count / count).item() for i, count in enumerate(class_counts)}
        self.lambda_base = 0.1  # You can tune this value
        self.class_weights = torch.tensor([
            self.lambda_base * math.log(1 + self.ir_ratios[i])
            for i in range(len(class_counts))
        ], dtype=torch.float32, device=config.device)

    def gradient_penalty(self, real_samples, fake_samples):
        alpha = torch.rand(real_samples.size(0), 1, device=config.device)
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates, _ = self.cd(interpolates)
        fake = torch.ones(d_interpolates.size(), device=config.device, requires_grad=False)
        
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True, 
        )[0]
        
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def fit(self, dataset):
        self.cd.train()
        for i in self.generators:
            i.train()
        self.divideSamples(dataset)
        cd_optimizer = torch.optim.Adam(self.cd.parameters(), lr=config.GAN_config.cd_lr, betas=(0.5, 0.9))
        g_optimizers = [
            torch.optim.Adam(g.parameters(), lr=config.GAN_config.g_lr, betas=(0.5, 0.9))
            for g in self.generators
        ]
        
        # Add Learning Rate Schedulers
        cd_scheduler = MultiStepLR(cd_optimizer, milestones=[1000, 1500], gamma=0.5)
        g_schedulers = [MultiStepLR(optimizer, milestones=[1000, 1500], gamma=0.5) for optimizer in g_optimizers]

        for e in range(config.GAN_config.epochs):
            print(f'\r{(e + 1) / config.GAN_config.epochs: .2%}', end='')
            
            epoch_d_loss = 0.0
            epoch_g_loss = 0.0
            epoch_c_loss_real = 0.0
            epoch_c_loss_fake = 0.0
            batch_count = 0

            for target_label in self.samples.keys():
                # --- Train Critic/Classifier ---
                for _ in range(config.GAN_config.cd_loopNo):
                    cd_optimizer.zero_grad()
                    
                    real_samples = self.get_target_samples(target_label, config.GAN_config.batch_size)
                    z = torch.randn(config.GAN_config.batch_size, config.GAN_config.z_size, device=config.device)
                    fake_samples = self.generators[target_label](z)

                    # WGAN Critic Loss
                    real_validity, _ = self.cd(real_samples)
                    fake_validity, _ = self.cd(fake_samples.detach())
                    gp = self.gradient_penalty(real_samples, fake_samples)
                    d_loss_wgan = -torch.mean(real_validity) + torch.mean(fake_validity) + 10 * gp
                    
                    # Classification loss for real data
                    _, real_logits = self.cd(real_samples)
                    real_labels = torch.full((config.GAN_config.batch_size,), target_label, dtype=torch.long, device=config.device)
                    c_loss_real = cross_entropy(real_logits, real_labels, weight=self.class_weights)
                    
                    # Total Discriminator Loss (with weights)
                    d_loss = self.lambda_adv * d_loss_wgan + self.lambda_cls * c_loss_real
                    d_loss.backward()
                    cd_optimizer.step()

                # --- Train Generator ---
                g_optimizers[target_label].zero_grad()

                z = torch.randn(config.GAN_config.batch_size, config.GAN_config.z_size, device=config.device)
                gen_samples = self.generators[target_label](z)
                
                # WGAN Adversarial Loss
                fake_validity, fake_logits = self.cd(gen_samples)
                g_loss_adv = -torch.mean(fake_validity)

                # Classification loss for fake data
                target_labels = torch.full((config.GAN_config.batch_size,), target_label, dtype=torch.long, device=config.device)
                c_loss_fake = cross_entropy(fake_logits, target_labels, weight=self.class_weights)

                # Feature Matching Loss
                with torch.no_grad():
                    real_samples = self.get_target_samples(target_label, config.GAN_config.batch_size)
                    real_features = self.cd.main_model(real_samples)
                fake_features = self.cd.main_model(gen_samples)
                fm_loss = torch.nn.functional.mse_loss(fake_features.mean(dim=0), real_features.mean(dim=0))

                # Total Generator Loss (with weights)
                g_loss = self.lambda_adv * g_loss_adv + self.lambda_cls * c_loss_fake + self.lambda_fm * fm_loss
                g_loss.backward()
                g_optimizers[target_label].step()

                # --- Logging ---
                epoch_d_loss += d_loss.item()
                epoch_g_loss += g_loss.item()
                epoch_c_loss_real += c_loss_real.item()
                epoch_c_loss_fake += c_loss_fake.item()
                batch_count += 1
            
            # Step the schedulers
            cd_scheduler.step()
            for scheduler in g_schedulers:
                scheduler.step()

            self.tracker.log_epoch(
                e + 1, 
                epoch_d_loss / batch_count, 
                epoch_g_loss / batch_count,
                epoch_c_loss_real / batch_count,
                epoch_c_loss_fake / batch_count
            )

            if (e + 1) in [500, 1000, 1500, 2000]:
                self.visualize_gan_output(e + 1, config.path_config.dpl_tmg_gan_out)
        
        self.tracker.plot_metrics(config.path_config.dpl_tmg_gan_out)
        self.cd.eval()
        for i in self.generators:
            i.eval()
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

    def visualize_gan_output(self, epoch, img_dir):
        with torch.no_grad():
            for i in range(datasets.label_num):
                real = self.get_target_samples(i, 100).cpu().numpy()
                gen = self.generate_samples(i, 100).cpu().numpy()
                plot_class_tsne(real, gen, i, epoch, img_dir)
            for i in self.generators:
                i.train() 