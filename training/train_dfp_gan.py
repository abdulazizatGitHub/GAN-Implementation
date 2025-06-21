# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# import torch
# import numpy as np
# from pathlib import Path
# import logging
# from datetime import datetime

# from training import config, datasets
# from training.DFP_GAN import DFPGAN
# from training.tracker import TrainingTracker
# from scripts.visualize_gan_output import Visualizer

# def setup_logging():
#     """Setup logging configuration"""
#     log_dir = Path("logs")
#     log_dir.mkdir(exist_ok=True)
    
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     log_file = log_dir / f"dfp_gan_training_{timestamp}.log"
    
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s - %(levelname)s - %(message)s',
#         handlers=[
#             logging.FileHandler(log_file),
#             logging.StreamHandler()
#         ]
#     )
#     return logging.getLogger(__name__)

# def main():
#     # Setup logging
#     logger = setup_logging()
#     logger.info("Starting DFP-GAN training on UNSW-NB15 dataset")
    
#     # Set random seeds for reproducibility
#     torch.manual_seed(42)
#     np.random.seed(42)
    
#     # Load and preprocess UNSW-NB15 dataset
#     logger.info("Loading UNSW-NB15 dataset...")
#     dataset = datasets.TrDataset()
#     logger.info(f"Dataset loaded with {len(dataset)} samples and {datasets.label_num} classes")
    
#     # Initialize DFP-GAN
#     logger.info("Initializing DFP-GAN...")
#     dfp_gan = DFPGAN()
    
#     # Create output directories
#     output_dir = Path("output/dfp_gan")
#     output_dir.mkdir(parents=True, exist_ok=True)
    
#     # Initialize visualizer
#     visualizer = Visualizer(output_dir)
    
#     try:
#         # Train DFP-GAN
#         logger.info("Starting training...")
#         dfp_gan.fit(dataset)
        
#         # Generate and save samples
#         logger.info("Generating samples...")
#         for label in range(datasets.label_num):
#             samples = dfp_gan.generate_qualified_samples(label, num=1000)
#             np.save(output_dir / f"generated_samples_class_{label}.npy", samples.cpu().numpy())
            
#             # Visualize samples
#             visualizer.visualize_samples(samples, label, f"dfp_gan_class_{label}")
        
#         # Save trained models
#         logger.info("Saving trained models...")
#         save_dir = Path("models/dfp_gan")
#         save_dir.mkdir(parents=True, exist_ok=True)
        
#         # Save CD model
#         torch.save(dfp_gan.cd.state_dict(), save_dir / "cd_model.pth")
        
#         # Save Generator models
#         for i, generator in enumerate(dfp_gan.generators):
#             torch.save(generator.state_dict(), save_dir / f"generator_{i}.pth")
        
#         logger.info("Training completed successfully!")
        
#     except Exception as e:
#         logger.error(f"An error occurred during training: {str(e)}", exc_info=True)
#         raise

# if __name__ == "__main__":
#     main() 