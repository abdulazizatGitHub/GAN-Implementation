import matplotlib.pyplot as plt
from training.logger import Logger

class TrainingTracker:
    def __init__(self, log_name="TMG_GAN_Training"):
        self.epochs = []
        self.d_losses = []
        self.g_losses = []
        self.c_losses = []
        self.inter_cosine = []
        self.intra_cosine = []
        self.logger = Logger(log_name)
        self.logger.turn_on()

    def log_epoch(self, epoch, d_loss, g_loss, c_loss, inter_cos, intra_cos):
        self.epochs.append(epoch)
        self.d_losses.append(d_loss)
        self.g_losses.append(g_loss)
        self.c_losses.append(c_loss)
        self.inter_cosine.append(inter_cos)
        self.intra_cosine.append(intra_cos)
        self.logger.info(
            f"Epoch {epoch}: D_loss={d_loss:.4f}, G_loss={g_loss:.4f}, C_loss={c_loss:.4f}, "
            f"Inter-cos={inter_cos:.4f}, Intra-cos={intra_cos:.4f}"
        )

    def plot_metrics(self, out_dir):
        # Losses
        plt.figure(figsize=(10, 6))
        plt.plot(self.epochs, self.d_losses, label='Discriminator Loss')
        plt.plot(self.epochs, self.g_losses, label='Generator Loss')
        plt.plot(self.epochs, self.c_losses, label='Classifier Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('TMG-GAN Training Losses')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(out_dir / "losses.png")
        plt.close()

        # Cosine Similarities
        plt.figure(figsize=(10, 6))
        plt.plot(self.epochs, self.inter_cosine, label='Inter-class Cosine Similarity')
        plt.plot(self.epochs, self.intra_cosine, label='Intra-class Cosine Similarity')
        plt.xlabel('Epoch')
        plt.ylabel('Cosine Similarity')
        plt.title('TMG-GAN Cosine Similarities')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(out_dir / "cosine_similarities.png")
        plt.close() 