import torch
import random
from torch.nn.functional import binary_cross_entropy
from training import config, datasets, models


class GAN:

    def __init__(self):
        self.generator = [
            models.GeneratorModel(config.GAN_config.z_size, datasets.feature_num).to(config.device)
            for _ in range(datasets.label_num)      
        ]

        self.discriminator = [
            models.DiscriminatorModel(datasets.feature_num,).to(config.device)
            for _ in range(datasets.label_num)
        ]
        self.samples = dict()


    def fit(self, dataset):
        for i in self.generator:
            i.train()

        for i in self.discriminator:
            i.train()

        self.divideSamples(dataset)

        d_optimizers = [
            torch.optim.Adam(
                params = self.discriminator[i].parameters(),
                lr = config.GAN_config.cd_lr,
                betas = (0.5, 0.999,)
            )
            for i in range(datasets.label_num)
        ]

        g_optimizers = [
            torch.optim.Adam(
                params = self.generator[i].parameters(),
                lr = config.GAN_config.g_lr,
                betas = (0.5, 0.999),
            )
            for i in range(datasets.label_num)
        ]

        for e in range(config.GAN_config.epochs):
            print(f'\r{(e + 1) / config.GAN_config.epochs: 0.2%}', end='')

            for target_label in range(datasets.label_num):
                #train D
                for _ in range(config.GAN_config.cd_loopNo):
                    d_optimizers[target_label].zero_grad()
                    real_samples = self.get_target_samples(target_label, config.GAN_config.batch_size)
                    score_real = self.discriminator[target_label](real_samples)
                    loss_real = binary_cross_entropy(score_real, torch.ones_like(score_real))
                    generated_samples = self.generator[target_label].generate_samples(config.GAN_config.batch_size)
                    score_generated = self.discriminator[target_label](generated_samples)
                    loss_generated = binary_cross_entropy(
                        score_generated,
                        torch.zeros_like(score_generated, device=config.device)
                    )
                    d_loss = (loss_real + loss_generated) / 2
                    d_loss.backward()
                    d_optimizers[target_label].step()

                #train G
                for _ in range(config.GAN_config.g_loopNo):
                    g_optimizers[target_label].zero_grad()
                    generated_samples = self.generator[target_label].generate_samples(config.GAN_config.batch_size)
                    score_generated = self.discriminator[target_label](generated_samples)
                    g_loss = binary_cross_entropy(
                        score_generated,
                        torch.ones_like(score_generated, device=config.device)
                    )
                    g_loss.backward()
                    g_optimizers[target_label].step()

        print()

        for i in self.discriminator:
            i.eval()

        for i in self.generator:
            i.eval()
    
    def divideSamples(self, dataset = datasets.TrDataset) -> None:
        for sample, label in dataset:
            lable = lable.item()
            
            if lable not in self.samples.keys():
                self.samples[label] = sample.unsqueeze(0)
            else:
                self.samples[label] = torch.cat([self.samples[label], sample.unsqeeze(0)])

    def get_target_samples(self, label: int, num: int) -> torch.Tensor:
        return torch.stack(
            random.choices(
                self.samples[label],
                k=num
            )
        )
    
    def generate_samples(self, target_label: int, num: int):
        return self.generator[target_label].generate_samples(num).cpu().detach()
