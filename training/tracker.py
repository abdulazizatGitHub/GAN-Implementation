import matplotlib.pyplot as plt
from training.logger import Logger

class TrainingTracker:
    def __init__(self, log_name="TMG_GAN_Training"):
        self.epochs = []
        self.d_losses = []
        self.g_losses = []
        self.c_losses_real = []
        self.c_losses_fake = []
        self.logger = Logger(log_name)
        self.logger.turn_on()

    def log_epoch(self, epoch, d_loss, g_loss, c_loss_real, c_loss_fake):
        self.epochs.append(epoch)
        self.d_losses.append(d_loss)
        self.g_losses.append(g_loss)
        self.c_losses_real.append(c_loss_real)
        self.c_losses_fake.append(c_loss_fake)
        self.logger.info(
            f"Epoch {epoch}: D_Loss={d_loss:.4f}, G_Loss={g_loss:.4f}, C_Loss_Real={c_loss_real:.4f}, C_Loss_Fake={c_loss_fake:.4f}"
        )

    def plot_metrics(self, out_dir):
        plt.figure(figsize=(12, 6))
        
        plt.plot(self.epochs, self.d_losses, label='Discriminator Total Loss')
        plt.plot(self.epochs, self.g_losses, label='Generator Total Loss')
        # plt.plot(self.epochs, self.c_losses_real, label='Classifier Loss (Real)', linestyle='--')
        # plt.plot(self.epochs, self.c_losses_fake, label='Classifier Loss (Fake)', linestyle='--')

        plt.title('Training Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        file_path = out_dir / "training_metrics.png"
        plt.savefig(file_path)
        plt.close() 