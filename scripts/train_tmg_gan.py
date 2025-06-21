import context

import pickle

import torch

import training
from training import Classifier,  datasets, utils
from training.TMG_GAN import TMGGAN as OriginalTMGGAN
from training.TMG_GAN_Dynamic import TMGGANDynamic

dataset = 'UNSW_NB15_training-set.csv'
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
    
    # Store original training samples and labels for independent balancing
    original_tr_samples = datasets.tr_samples.clone()
    original_tr_labels = datasets.tr_labels.clone()
    original_te_samples = datasets.te_samples.clone()
    original_te_labels = datasets.te_labels.clone()

    print("\n--- Training Original TMG_GAN ---")
    original_tmg_gan = OriginalTMGGAN()
    
    # Create a dataset instance for original GAN (using a copy of original data)
    current_tr_dataset_original_gan = training.datasets.TrDataset()
    current_tr_dataset_original_gan.samples = original_tr_samples.clone()
    current_tr_dataset_original_gan.labels = original_tr_labels.clone()
    original_tmg_gan.fit(current_tr_dataset_original_gan)

    print("\n--- Generating samples from Original TMG_GAN for visualization ---")
    original_gen_samples = []
    with torch.no_grad():
        for i in range(datasets.label_num):
            # Generate 100 samples per class from the original GAN
            original_gen_samples.append(original_tmg_gan.generate_samples(i, 100).cpu().numpy())

    # --- Save balanced dataset from Original TMG_GAN ---
    max_cnt_orig = max([len(original_tmg_gan.samples[i]) for i in original_tmg_gan.samples.keys()])
    tr_samples_balanced_orig = original_tr_samples.clone() # Start with a fresh copy of original data
    tr_labels_balanced_orig = original_tr_labels.clone()

    for i in original_tmg_gan.samples.keys():
        cnt_generated = max_cnt_orig - len(original_tmg_gan.samples[i])
        if cnt_generated > 0:
            generated_samples = original_tmg_gan.generate_qualified_samples(i, cnt_generated)
            generated_labels = torch.full([cnt_generated], i)
            tr_samples_balanced_orig = torch.cat([tr_samples_balanced_orig, generated_samples])
            tr_labels_balanced_orig = torch.cat([tr_labels_balanced_orig, generated_labels])

    with open('data/original_tmg_gan_balanced_dataset.pkl', 'wb') as f:
        pickle.dump(
            (
                tr_samples_balanced_orig.numpy(),
                tr_labels_balanced_orig.numpy(),
                original_te_samples.numpy(), # Test samples remain unchanged
                original_te_labels.numpy(),  # Test labels remain unchanged
            ),
            f,
        )
    print("Original TMG_GAN balanced dataset saved to data/original_tmg_gan_balanced_dataset.pkl")

    print("\n--- Training Dynamic TMG_GAN ---")
    dynamic_tmg_gan = TMGGANDynamic()
    
    # Create a dataset instance for dynamic GAN (using a fresh copy of original data)
    current_tr_dataset_dynamic_gan = training.datasets.TrDataset()
    current_tr_dataset_dynamic_gan.samples = original_tr_samples.clone()
    current_tr_dataset_dynamic_gan.labels = original_tr_labels.clone()
    dynamic_tmg_gan.fit(current_tr_dataset_dynamic_gan, original_gan_samples=original_gen_samples)
    
    # --- Save balanced dataset from Dynamic TMG_GAN ---
    max_cnt_dynamic = max([len(dynamic_tmg_gan.samples[i]) for i in dynamic_tmg_gan.samples.keys()])
    tr_samples_balanced_dynamic = original_tr_samples.clone() # Start with a fresh copy of original data
    tr_labels_balanced_dynamic = original_tr_labels.clone()

    for i in dynamic_tmg_gan.samples.keys():
        cnt_generated = max_cnt_dynamic - len(dynamic_tmg_gan.samples[i])
        if cnt_generated > 0:
            generated_samples = dynamic_tmg_gan.generate_qualified_samples(i, cnt_generated)
            generated_labels = torch.full([cnt_generated], i)
            tr_samples_balanced_dynamic = torch.cat([tr_samples_balanced_dynamic, generated_samples])
            tr_labels_balanced_dynamic = torch.cat([tr_labels_balanced_dynamic, generated_labels])

    with open('data/dynamic_tmg_gan_balanced_dataset.pkl', 'wb') as f:
        pickle.dump(
            (
                tr_samples_balanced_dynamic.numpy(),
                tr_labels_balanced_dynamic.numpy(),
                original_te_samples.numpy(), # Test samples remain unchanged
                original_te_labels.numpy(),  # Test labels remain unchanged
            ),
            f,
        )
    print("Dynamic TMG_GAN balanced dataset saved to data/dynamic_tmg_gan_balanced_dataset.pkl")

    torch.cuda.empty_cache()

    print("\nTraining and visualization complete!")

    # Commented out classification section for now, as datasets.tr_samples and datasets.tr_labels
    # are no longer globally modified in a way that directly supports this original structure.
    # If classification is needed, load the specific balanced dataset from the pkl file.
    # utils.set_random_state()
    # clf = Classifier('TMG_GAN')
    # clf.model = tmg_gan.cd
    # clf.fit(datasets.TrDataset())