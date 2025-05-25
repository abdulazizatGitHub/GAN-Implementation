import random
import torch
from torch.nn.functional import cross_entropy, cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from training import config, datasets, models
from scripts.visualize_gan_output import plot_gan_output_grid

class TMGGAN:

    def __init__(self):
        self.generators = [
            models.GeneratorModel(config.GAN_config.z_size, datasets.feature_num).to(config.device)
            for _ in range (datasets.label_num)
        ]   

        self.cd = models.CDModel(datasets.feature_num, datasets.label_num).to(config.device)

        self.samples = dict()

    def fit(self, dataset):
        self.g_losses = []
        self.d_losses = []
        self.c_losses = []
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

            epoch_g_loss = 0.0
            epoch_d_loss = 0.0
            epoch_c_loss = 0.0
            batch_count = 0

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
                    
                    g_loss = -score_generated + loss_label + cd_hidden_loss
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
            
            g_hidden_losses = torch.mean(torch.stack(g_hidden_losses)) / datasets.feature_num
            g_hidden_losses.backward()

            for i in g_optimizers:
                i.step()
            
            if e % 10 == 0:
                self.visualize_generated_samples(e)

            # Store average losses for the epoch
            if batch_count > 0:
                self.g_losses.append(epoch_g_loss / batch_count)
                self.d_losses.append(epoch_d_loss / batch_count)
                self.c_losses.append(epoch_c_loss / batch_count)
            else:
                self.g_losses.append(0.0)
                self.d_losses.append(0.0)
                self.c_losses.append(0.0)
        
        print('')
        self.cd.eval()

        for i in self.generators:
            i.eval()

        # Save losses to a .log file
        with open('data/logs/loss_history.log', 'w') as f:
            for epoch, (g, d, c) in enumerate(zip(self.g_losses, self.d_losses, self.c_losses), 1):
                f.write(f'Epoch {epoch}: G_loss={g:.4f}, D_loss={d:.4f}, C_loss={c:.4f}\n')

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
                real_samples, generated_samples, epoch,
                config.path_config.GAN_out, n_classes=datasets.label_num, n_points=100
            )
            
            for i in self.generators:
                i.train()
