# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# import training
# from training import datasets, utils, tracker


# if __name__ == '__main__':
#     utils.set_random_state()
    
#     train_dataset = datasets.TrDataset()

#     t_gan = training.TGAN()
#     t_gan.divideSamples(train_dataset)  # Prepares real samples
    
#     t_gan.fit()  # Starts training, no dataset needed here
#     t_gan.tracker.plot_losses()
#     t_gan.tracker.plot_cosine_similarities()


