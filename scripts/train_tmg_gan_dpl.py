import context

import pickle

import torch

import training
from training import Classifier, datasets, utils

from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# dataset = 'KDDCUP99'
# dataset = 'NSL-KDD'

if __name__ == '__main__':
    # utils.set_random_state()
    # utils.prepare_datasets(dataset)
    # utils.turn_on_test_mode()
    # utils.transfer_to_binary()

    # # select features
    # lens = (len(datasets.tr_samples), len(datasets.te_samples))
    # samples = torch.cat(
    #     [
    #         datasets.tr_samples,
    #         datasets.te_samples,
    #     ]
    # )
    # labels = torch.cat(
    #     [
    #         datasets.tr_labels,
    #         datasets.te_labels,
    #     ]
    # )
    # from sklearn.decomposition import PCA
    # from sklearn.preprocessing import minmax_scale
    #
    # pca = PCA(n_components=25)
    # samples = torch.from_numpy(
    #     minmax_scale(
    #         pca.fit_transform(samples, labels)
    #     )
    # ).float()
    # samples = (samples - samples.min())
    # datasets.tr_samples, datasets.te_samples = torch.split(samples, lens)
    # utils.set_dataset_values()
    # print(datasets.feature_num)

    utils.set_random_state()
    tmg_gan = training.MYMETHOD()
    tmg_gan.fit(training.datasets.TrDataset())
    # count the max number of samples
    max_cnt = max([len(tmg_gan.samples[i]) for i in tmg_gan.samples.keys()])

    # Compute class weights
    labels = np.array([label for _, label in datasets.TrDataset()])
    classes = np.unique(labels)
    class_weights = compute_class_weight('balanced', classes=classes, y=labels)
    # Normalize class weights so the largest is 1
    norm_class_weights = class_weights / class_weights.max()

    # generate samples
    for i, w in enumerate(norm_class_weights):
        cnt_generated = int((max_cnt - len(tmg_gan.samples[i])) * w)
        if cnt_generated > 0:
            generated_samples = tmg_gan.generate_qualified_samples(i, cnt_generated)
            generated_labels = torch.full([cnt_generated], i)
            datasets.tr_samples = torch.cat([datasets.tr_samples, generated_samples])
            datasets.tr_labels = torch.cat([datasets.tr_labels, generated_labels])

    with open('dpl_data.pkl', 'wb') as f:
        pickle.dump(
            (
                datasets.tr_samples.numpy(),
                datasets.tr_labels.numpy(),
                datasets.te_samples.numpy(),
                datasets.te_labels.numpy(),
            ),
            f,
        )

    # utils.set_random_state()
    # clf = Classifier('TMG_GAN')
    # clf.model = tmg_gan.cd
    # clf.fit(datasets.TrDataset())
    # torch.cuda.empty_cache()
    # clf.test(datasets.TeDataset())
    # print(clf.confusion_matrix)
    # print(clf.metrics)
    # clf.binary_test(datasets.TeDataset())
    # print(clf.confusion_matrix)
    # print(clf.metrics)