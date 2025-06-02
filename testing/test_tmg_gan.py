import context

import torch

import training

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

if __name__ == '__main__':
    training.utils.set_random_state()
    
    # Print detailed preprocessing information
    print("\nPreprocessing Information:")
    print(f"Feature dimension from datasets: {training.datasets.feature_num}")
    print(f"Test samples shape: {training.datasets.te_samples.shape}")
    print(f"Number of numerical features: {len(training.datasets.numerical_columns)}")
    print(f"Number of categorical features: {len(training.datasets.categorical_columns)}")
    print(f"Preprocessor transformers: {training.datasets.preprocessor.transformers_}")
    
    # Initialize TMG-GAN with correct feature dimension
    tmg_gan = training.TMGGAN()
    
    # Load saved models
    model_dir = training.config.path_config.data / "trained_models"
    
    # Load CD model
    cd_state = torch.load(model_dir / "cd_model.pth")
    # Update feature dimension in state dict if needed
    if cd_state['main_model.0.parametrizations.weight.original'].shape[1] != training.datasets.feature_num:
        print(f"\nWarning: Model expects {cd_state['main_model.0.parametrizations.weight.original'].shape[1]} features, but data has {training.datasets.feature_num} features")
        print("This suggests a mismatch between the preprocessing used during training and testing.")
        print("Please ensure you're using the same preprocessing method (OneHotEncoder vs OrdinalEncoder) as during training.")
        raise ValueError("Feature dimension mismatch between saved model and current data")
    
    tmg_gan.cd.load_state_dict(cd_state)
    tmg_gan.cd.eval()  # Set to evaluation mode
    
    # Load Generator models
    for i, generator in enumerate(tmg_gan.generators):
        gen_state = torch.load(model_dir / f"generator_{i}.pth")
        # Update feature dimension in state dict if needed
        if gen_state['last_layer.0.weight'].shape[0] != training.datasets.feature_num:
            print(f"\nWarning: Generator expects {gen_state['last_layer.0.weight'].shape[0]} features, but data has {training.datasets.feature_num} features")
            print("This suggests a mismatch between the preprocessing used during training and testing.")
            print("Please ensure you're using the same preprocessing method (OneHotEncoder vs OrdinalEncoder) as during training.")
            raise ValueError("Feature dimension mismatch between saved model and current data")
        
        tmg_gan.generators[i].load_state_dict(gen_state)
        tmg_gan.generators[i].eval()  # Set to evaluation mode
    
    print("\nLoaded saved models successfully")

    # Get preprocessed test data and labels
    x_real = training.datasets.te_samples.cpu()
    y_real = training.datasets.te_labels.cpu()

    # Count samples per class in test set
    class_counts = torch.bincount(y_real, minlength=training.datasets.label_num)
    num_generated_per_class = class_counts.tolist()

    n_per_class = 1000  # Number of real and generated samples per class

    x_generated_list = []
    y_generated_list = []
    x_real_list = []
    y_real_list = []

    for i in range(training.datasets.label_num):
        # Get all real samples for this class
        real_mask = (y_real == i)
        real_data = x_real[real_mask]
        # If there are more than n_per_class, randomly select n_per_class
        if len(real_data) > n_per_class:
            idx = torch.randperm(len(real_data))[:n_per_class]
            real_data = real_data[idx]
        # If there are fewer, use all available
        x_real_list.append(real_data)
        y_real_list.append(torch.full([len(real_data)], i, dtype=torch.long))

        # Generate exactly n_per_class samples for this class
        gen_samples = tmg_gan.generate_qualified_samples(i, n_per_class)
        x_generated_list.append(gen_samples)
        y_generated_list.append(torch.full([n_per_class], i, dtype=torch.long))

    x_real = torch.cat(x_real_list)
    y_real = torch.cat(y_real_list)
    x_generated = torch.cat(x_generated_list)
    y_generated = torch.cat(y_generated_list)

    # Concatenate real and generated data and labels
    x_combined = torch.cat([x_real, x_generated])
    y_combined = torch.cat([y_real, y_generated])

    # Create a source label array (Real/Generated)
    source_combined = ['Real'] * len(x_real) + ['Generated'] * len(x_generated)

    # Convert to numpy for TSNE
    x_combined_np = x_combined.numpy()
    y_combined_np = y_combined.numpy()
    source_combined_np = np.array(source_combined)

    # Create combined labels for plotting (class.source_id)
    # Map 'Real' to 0 and 'Generated' to 1
    source_id_map = {'Real': 0, 'Generated': 1}
    combined_labels = [f'{y}.{source_id_map[source]}' for y, source in zip(y_combined_np, source_combined_np)]

    # --- t-SNE Visualization: Per-class grid, matching training style ---
    n_classes = training.datasets.label_num
    fig, axes = plt.subplots(1, n_classes, figsize=(3 * n_classes, 3))
    if n_classes == 1:
        axes = [axes]

    for i in range(n_classes):
        ax = axes[i]
        # Get real and generated samples for class i
        real_mask = (y_real == i)
        gen_mask = (y_generated == i)
        real_data = x_real[real_mask].numpy()
        gen_data = x_generated[gen_mask].numpy()

        # Combine for t-SNE
        combined_data = np.concatenate([real_data, gen_data], axis=0)
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        combined_proj = tsne.fit_transform(combined_data)

        real_proj = combined_proj[:len(real_data)]
        gen_proj = combined_proj[len(real_data):]

        ax.scatter(real_proj[:, 0], real_proj[:, 1], color='blue', s=10, label='Real', alpha=0.6)
        ax.scatter(gen_proj[:, 0], gen_proj[:, 1], color='orange', s=10, label='Generated', alpha=0.6)
        ax.set_title(f'Class {i}')
        ax.set_xticks([])
        ax.set_yticks([])
        if i == 0:
            ax.legend()

    plt.tight_layout()
    plt.savefig('testing/tmg_gan_test_grid_by_class_tsne.png')
    plt.show()