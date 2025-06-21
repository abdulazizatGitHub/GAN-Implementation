import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.manifold import TSNE

# Only plotting utility at top-level

def plot_gan_output_grid(real_samples, original_gan_samples, dynamic_gan_samples, epoch, out_path, n_classes=5, n_points=100):
    """
    Plots a grid of scatter plots comparing real, original GAN, and dynamically weighted GAN generated samples for each class.
    Args:
        real_samples: list of np.arrays, each [N, D] for each class (blue)
        original_gan_samples: list of np.arrays, each [N, D] for each class (orange)
        dynamic_gan_samples: list of np.arrays, each [N, D] for each class (red)
        epoch: int, current epoch
        out_path: Path or str, where to save the image
        n_classes: int, number of classes
        n_points: int, This parameter is no longer strictly used for slicing,
                  the function will plot all samples provided in the input lists.
    """
    fig, axes = plt.subplots(1, n_classes, figsize=(3*n_classes, 3))
    if n_classes == 1:
        axes = [axes]
    for i in range(n_classes):
        ax = axes[i]
        
        real_data = real_samples[i]
        orig_gen_data = original_gan_samples[i]
        dynamic_gen_data = dynamic_gan_samples[i]

        if len(real_data) > 0 or len(orig_gen_data) > 0 or len(dynamic_gen_data) > 0:
            # Combine data for t-SNE fitting
            combined_data = np.concatenate([real_data, orig_gen_data, dynamic_gen_data], axis=0)
            # Use t-SNE for 2D projection
            tsne = TSNE(n_components=2, perplexity=30, random_state=42)
            combined_proj = tsne.fit_transform(combined_data)
            
            # Split projected data back
            real_proj = combined_proj[:len(real_data)]
            orig_gen_proj = combined_proj[len(real_data):len(real_data) + len(orig_gen_data)]
            dynamic_gen_proj = combined_proj[len(real_data) + len(orig_gen_data):]
            
            ax.scatter(real_proj[:, 0], real_proj[:, 1], color='blue', s=10, label='Real', alpha=0.6)
            ax.scatter(orig_gen_proj[:, 0], orig_gen_proj[:, 1], color='orange', s=10, label='Original GAN Generated', alpha=0.6)
            ax.scatter(dynamic_gen_proj[:, 0], dynamic_gen_proj[:, 1], color='red', s=10, label='Dynamic GAN Generated', alpha=0.6)
        
        ax.set_title(f'Class {i}')
        ax.set_xticks([])
        ax.set_yticks([])
        if i == 0:
            ax.legend()
    plt.tight_layout()
    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path / f'gan_output_epoch_{epoch}.png')
    plt.close(fig)

# --- New: Feature-wise visualization ---
def plot_featurewise_histograms(real_samples, generated_samples, epoch, out_path, n_classes=5, n_features=5):
    """
    Plots histograms for the first n_features of real vs generated samples for each class.
    """
    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)
    for class_idx in range(n_classes):
        real = real_samples[class_idx]
        gen = generated_samples[class_idx]
        fig, axes = plt.subplots(1, n_features, figsize=(4*n_features, 3))
        for f in range(n_features):
            ax = axes[f]
            ax.hist(real[:, f], bins=30, alpha=0.6, color='blue', label='Real')
            ax.hist(gen[:, f], bins=30, alpha=0.6, color='orange', label='Generated')
            ax.set_title(f'Feature {f}')
            if f == 0:
                ax.legend()
        plt.suptitle(f'Class {class_idx} - Feature-wise Histograms (Epoch {epoch})')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(out_path / f'feature_hist_class{class_idx}_epoch{epoch}.png')
        plt.close(fig)

# --- New: Feature-wise scatter plot (first two features) ---
def plot_featurewise_scatter(real_samples, generated_samples, epoch, out_path, n_classes=5):
    """
    Plots scatter plots of the first two features for real vs generated samples for each class.
    """
    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)
    for class_idx in range(n_classes):
        real = real_samples[class_idx]
        gen = generated_samples[class_idx]
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(real[:, 0], real[:, 1], color='blue', s=10, label='Real', alpha=0.6)
        ax.scatter(gen[:, 0], gen[:, 1], color='orange', s=10, label='Generated', alpha=0.6)
        ax.set_title(f'Class {class_idx} - Feature 0 vs Feature 1 (Epoch {epoch})')
        ax.set_xlabel('Feature 0')
        ax.set_ylabel('Feature 1')
        ax.legend()
        plt.tight_layout()
        plt.savefig(out_path / f'feature_scatter_class{class_idx}_epoch{epoch}.png')
        plt.close(fig)

# Demo/test code can go here, but should not run on import
if __name__ == '__main__':
    # Example usage or test code can be placed here if needed
    pass