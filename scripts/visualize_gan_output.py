import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA

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
        n_points: int, number of points to plot per class
    """
    fig, axes = plt.subplots(1, n_classes, figsize=(3*n_classes, 3))
    if n_classes == 1:
        axes = [axes]
    for i in range(n_classes):
        ax = axes[i]
        # Use PCA for 2D projection if needed
        pca = PCA(n_components=2)
        real_proj = pca.fit_transform(real_samples[i][:n_points])
        gen_proj = pca.transform(generated_samples[i][:n_points])
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