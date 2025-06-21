# import torch
# import argparse
# from pathlib import Path
# import sys
# import os

# # Add the project root to Python path
# project_root = Path(__file__).parent.parent
# sys.path.append(str(project_root))

# from training import config, datasets
# from training.HDSF_GAN import HDSFGAN
# from training.utils import setup_logging, load_config, set_random_state
# from training.tracker import TrainingTracker

# def parse_args():
#     parser = argparse.ArgumentParser(description='Train HDSF-GAN model on UNSW-NB15 dataset')
#     parser.add_argument('--train_path', type=str, 
#                       default='data/datasets/UNSW_NB15_training-set.csv',
#                       help='Path to training dataset CSV')
#     parser.add_argument('--test_path', type=str, 
#                       default='data/datasets/UNSW_NB15_testing-set.csv',
#                       help='Path to testing dataset CSV')
#     parser.add_argument('--output_dir', type=str, default='output/hdsf_gan',
#                       help='Directory to save model outputs')
#     parser.add_argument('--batch_size', type=int, default=64,
#                       help='Batch size for training')
#     parser.add_argument('--epochs', type=int, default=2000,
#                       help='Number of training epochs')
#     parser.add_argument('--g_lr', type=float, default=0.0002,
#                       help='Learning rate for generator')
#     parser.add_argument('--cd_lr', type=float, default=0.0002,
#                       help='Learning rate for CD model')
#     parser.add_argument('--z_size', type=int, default=100,
#                       help='Size of latent vector')
#     parser.add_argument('--g_loop', type=int, default=1,
#                       help='Number of generator training iterations per epoch')
#     parser.add_argument('--cd_loop', type=int, default=1,
#                       help='Number of CD model training iterations per epoch')
#     parser.add_argument('--seed', type=int, default=42,
#                       help='Random seed for reproducibility')
#     return parser.parse_args()

# def main():
#     # Parse command line arguments
#     args = parse_args()
    
#     # Setup logging
#     logger = setup_logging(args.output_dir)
#     logger.info("Starting HDSF-GAN training on UNSW-NB15 dataset")
    
#     # Set random seed
#     set_random_state(args.seed)
    
#     # Load configuration
#     config_path = project_root / 'config/default_config.yaml'
#     if config_path.exists():
#         load_config(config_path)
    
#     # Update config with command line arguments
#     config.GAN_config.batch_size = args.batch_size
#     config.GAN_config.epochs = args.epochs
#     config.GAN_config.g_lr = args.g_lr
#     config.GAN_config.cd_lr = args.cd_lr
#     config.GAN_config.z_size = args.z_size
#     config.GAN_config.g_loopNo = args.g_loop
#     config.GAN_config.cd_loopNo = args.cd_loop
    
#     # Create output directory
#     output_dir = Path(args.output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)
#     config.path_config.GAN_out = output_dir
    
#     # Set device
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     config.device = device
#     logger.info(f"Using device: {device}")
    
#     try:
#         # Load and preprocess dataset using existing preprocessing
#         logger.info("Loading and preprocessing UNSW-NB15 dataset...")
#         datasets.tr_samples, datasets.tr_labels, datasets.te_samples, datasets.te_labels, _ = datasets.load_and_preprocess_data(
#             args.train_path,
#             args.test_path
#         )
        
#         # Update dataset dimensions
#         datasets.feature_num = datasets.tr_samples.shape[1]
#         datasets.label_num = 5  # Number of classes in UNSW-NB15
        
#         # Calculate class weights for handling imbalance
#         class_counts = torch.bincount(datasets.tr_labels)
#         datasets.class_weights = 1.0 / class_counts
#         datasets.class_weights = datasets.class_weights / datasets.class_weights.sum()
        
#         logger.info(f"Dataset loaded with {len(datasets.tr_samples)} training samples")
#         logger.info(f"Feature dimensions: {datasets.feature_num}")
#         logger.info(f"Number of classes: {datasets.label_num}")
#         logger.info("Class distribution:")
#         for i, count in enumerate(class_counts):
#             logger.info(f"Class {i}: {count} samples")
        
#         # Initialize HDSF-GAN
#         logger.info("Initializing HDSF-GAN...")
#         model = HDSFGAN()
        
#         # Train model
#         logger.info("Starting training...")
#         model.fit(datasets.TrDataset())
        
#         # Save final model
#         logger.info("Saving final model...")
#         save_dir = output_dir / "final_model"
#         save_dir.mkdir(parents=True, exist_ok=True)
        
#         torch.save({
#             'cd_state_dict': model.cd.state_dict(),
#             'generators_state_dict': [g.state_dict() for g in model.generators],
#             'config': config.GAN_config.__dict__,
#             'feature_num': datasets.feature_num,
#             'label_num': datasets.label_num,
#             'class_weights': datasets.class_weights
#         }, save_dir / "hdsf_gan_final.pth")
        
#         logger.info("Training completed successfully!")
        
#     except Exception as e:
#         logger.error(f"An error occurred during training: {str(e)}")
#         raise

# if __name__ == "__main__":
#     main() 