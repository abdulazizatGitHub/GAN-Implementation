# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# import torch
# from training import config, datasets
# from training.adaptive_gan import AdaptiveGAN

# def main():
#     # Load and preprocess dataset
#     print("Loading and preprocessing dataset...")
#     dataset = datasets.TrDataset()
    
#     # Print dataset information
#     print("\nDataset Information:")
#     print(f"Number of samples: {len(dataset)}")
#     print(f"Number of features: {datasets.feature_num}")
#     print(f"Number of classes: {datasets.label_num}")
    
#     # Initialize model
#     print("\nInitializing Adaptive GAN...")
#     model = AdaptiveGAN(
#         z_size=config.GAN_config.z_size,
#         class_num=datasets.label_num,
#         feature_num=datasets.feature_num
#     )
    
#     # Train model
#     print("\nStarting training...")
#     model.fit(dataset)
    
#     print("\nTraining completed!")

# if __name__ == "__main__":
#     main() 