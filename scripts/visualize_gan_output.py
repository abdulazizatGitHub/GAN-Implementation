import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Only plotting utility at top-level

def plot_gan_output_grid(real_samples, generated_samples, epoch, out_path, n_classes=5, n_points=100):
    """
    Plots a grid of scatter plots comparing real and generated samples for each class.
    Args:
        real_samples: list of np.arrays, each [N, D] for each class
        generated_samples: list of np.arrays, each [N, D] for each class
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
        gen_data = generated_samples[i]

        if len(real_data) > 0 or len(gen_data) > 0:
            # Combine data for t-SNE fitting
            combined_data = np.concatenate([real_data, gen_data], axis=0)
            # Use t-SNE for 2D projection
            tsne = TSNE(n_components=2, perplexity=30, random_state=42)
            combined_proj = tsne.fit_transform(combined_data)
            
            # Split projected data back
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
    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path / f'gan_output_epoch_{epoch}.png')
    plt.close(fig)

# Demo/test code can go here, but should not run on import
if __name__ == '__main__':
    # Example usage or test code can be placed here if needed
    pass