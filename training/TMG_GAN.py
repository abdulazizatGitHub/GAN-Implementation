import random
import torch
from torch.nn.functional import cross_entropy, cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from training import config, datasets, models
from scripts.visualize_gan_output import plot_gan_output_grid
from training.tracker import TrainingTracker

class TMGGAN:

    def __init__(self):
        self.generators = [
            models.GeneratorModel(config.GAN_config.z_size, datasets.feature_num).to(config.device)
            for _ in range (datasets.label_num)
        ]   

        self.cd = models.CDModel(datasets.feature_num, datasets.label_num).to(config.device)

        self.samples = dict()
        self.tracker = TrainingTracker()

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
            inter_cosine_vals = []
            intra_cosine_vals = []
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
                    
                    # --- Cosine Similarity Loss (Equation 4) ---
                    # Features for generated samples of class k
                    F_gen_k = self.cd.main_model(generated_samples)
                    # Features for real samples of class k
                    F_real_k = self.cd.main_model(real_samples)
                    # Features for generated samples of all other classes
                    F_gen_j_list = []
                    for j in range(datasets.label_num):
                        if j == target_label:
                            continue
                        gen_j = self.generators[j].generate_samples(config.GAN_config.batch_size).to(config.device)
                        F_gen_j = self.cd.main_model(gen_j)
                        F_gen_j_list.append(F_gen_j)
                    F_gen_j_all = torch.cat(F_gen_j_list, dim=0)  # [num_other_gen, feat_dim]

                    # Normalize features
                    F_gen_k_norm = F_gen_k / F_gen_k.norm(dim=1, keepdim=True)
                    F_real_k_norm = F_real_k / F_real_k.norm(dim=1, keepdim=True)
                    F_gen_j_norm = F_gen_j_all / F_gen_j_all.norm(dim=1, keepdim=True)

                    # Intra similarity: mean cosine similarity between F_gen_k and F_real_k
                    intra_cos_matrix = torch.mm(F_gen_k_norm, F_real_k_norm.t()).abs()
                    intra_sim = intra_cos_matrix.mean()

                    # Inter similarity: mean cosine similarity between F_gen_k and F_gen_j_all
                    inter_cos_matrix = torch.mm(F_gen_k_norm, F_gen_j_norm.t()).abs()
                    inter_sim = inter_cos_matrix.mean()

                    cosine_loss = inter_sim - intra_sim

                    # --- Add to generator loss ---
                    g_loss = -score_generated + loss_label + cd_hidden_loss + cosine_loss
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
            
            if e in [100, 200, 300, 400, 499]:
                self.visualize_generated_samples(e)
        
        print('')
        self.cd.eval()

        for i in self.generators:
            i.eval()
        
        # Plot metrics at the end
        self.tracker.plot_metrics(config.path_config.GAN_out)
    

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
