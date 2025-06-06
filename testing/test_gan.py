import context

import torch

import training

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

if __name__ == '__main__':
    # src.utils.turn_on_test_mode()
    training.utils.prepare_datasets()
    training.utils.set_random_state()
    gan = training.GAN()
    gan.fit(training.datasets.TrDataset())

    x = training.datasets.TrDataset().samples.cpu()
    y = training.datasets.TrDataset().labels.cpu()

    for i in range(training.datasets.label_num):
        x = torch.cat([x, gan.generate_samples(i, len(gan.samples[i]))])
        y = torch.cat([y, torch.full([len(gan.samples[i])], i + 0.1)])

    embedded_x = TSNE(
        learning_rate='auto',
        init='random',
        random_state=training.config.seed,
    ).fit_transform(x)
    sns.scatterplot(
        x=embedded_x[:, 0],
        y=embedded_x[:, 1],
        hue=y,
        palette="deep",
        alpha=0.3,
    )
    plt.savefig('gan.jpg')
    plt.show()