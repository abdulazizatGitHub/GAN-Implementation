import numpy as np
import tensorflow as tf
import random
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import (classification_report, roc_auc_score, f1_score, recall_score, accuracy_score,
                             confusion_matrix, precision_score, roc_curve, auc)
import os

# Check if GPU is available
print(tf.config.list_physical_devices('GPU'))

# Load data
with open('dpl_data.pkl', 'rb') as f:
    tr_samples, tr_labels, te_samples, te_labels = pickle.load(f)

# Print the data shape to ensure it meets expectations
print(f"Training samples shape: {tr_samples.shape}")
print(f"Test samples shape: {te_samples.shape}")

# Dynamically determine input dimension
input_dim = tr_samples.shape[1] # Number of features
print(f"Dynamically determined input dimension: {input_dim}")

# One-hot encode the labels
training_labels = to_categorical(tr_labels)
test_labels = to_categorical(te_labels)

# Data used directly without reshaping for CNN (as it's a DNN classifier)
x_train = tr_samples
x_test = te_samples

# Print adjusted data shape
print(f"x_train prepared for DNN: {x_train.shape}")
print(f"x_test prepared for DNN: {x_test.shape}")

# Model parameters
n_hidden_1 = 256 # This parameter becomes less relevant as we are dynamically setting sizes
n_classes = 5
training_epochs = 50
batch_size = 128  # Reduce the batch size to decrease memory usage

# Create output folder
output_dir = os.path.join('./classification/TMG_WGAN_results/', 'multi_class')
os.makedirs(output_dir, exist_ok=True)

# Define sentiment category labels
EMOS = ['0', '1', '2', '3', '4']
NUM_EMO = len(EMOS)


# Function to create and train the model
def train_and_evaluate(seed):
    print(f"\nRunning with seed: {seed}")

    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Build the model
    model = Sequential([
        Dense(256, activation = 'relu', input_shape=(input_dim,)),
        Dense(128, activation = 'relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(n_classes, activation='softmax')
    ])

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    model.fit(x_train, training_labels, batch_size=batch_size, epochs=training_epochs, shuffle=True, verbose=1)

    # Model evaluation
    score = model.evaluate(x_test, test_labels, batch_size=batch_size, verbose=1)
    print(f"Test loss: {score[0]}, Test accuracy: {score[1]}")

    # Prediction
    pre = model.predict(x_test)

    # Convert prediction results to binary (0 or 1)
    pre_binary = (pre == pre.max(axis=1, keepdims=True)).astype(int)

    # Convert to single-class labels
    y_pred = np.argmax(pre_binary, axis=1)
    y_true = np.argmax(test_labels, axis=1)

    # Calculate evaluation metrics
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    auc_score = roc_auc_score(test_labels, pre, multi_class='ovo')

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    # Plot and save confusion matrix heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, cmap='Blues', fmt=".2f")
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title(f'Confusion Matrix (Seed: {seed})')

    # Save the heatmap as an image
    heatmap_path = os.path.join(output_dir, f'confusion_matrix_seed_{seed}.png')
    plt.savefig(heatmap_path)
    plt.close()

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc_dict = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(test_labels[:, i], pre[:, i])
        roc_auc_dict[i] = auc(fpr[i], tpr[i])

    # Plot ROC curves
    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], lw=2, label=f'ROC curve of class {EMOS[i]} (area = {roc_auc_dict[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (ROC) Curves (Seed: {seed})')
    plt.legend(loc="lower right")
    roc_plot_path = os.path.join(output_dir, f'roc_curves_seed_{seed}.png')
    plt.savefig(roc_plot_path)
    plt.close()

    # Print evaluation metrics
    print(f"Recall: {recall:.4f}, F1: {f1:.4f}, Accuracy: {acc:.4f}, Precision: {precision:.4f}, AUC: {auc_score:.4f}")

    # --- Per-seed metric bar plot ---
    metrics = [acc, recall, f1, precision]
    metric_names = ['Accuracy', 'Recall', 'F1 Score', 'Precision']
    plt.figure(figsize=(6, 4))
    plt.bar(metric_names, metrics, color=['blue', 'orange', 'green', 'red'])
    plt.ylim(0, 1)
    plt.title(f'Metrics for Seed {seed}')
    for i, v in enumerate(metrics):
        plt.text(i, v + 0.01, f'{v:.2f}', ha='center')
    plt.ylabel('Score')
    os.makedirs(os.path.join(output_dir, 'matrices_bar_plots'), exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'matrices_bar_plots', f'metrics_bar_seed_{seed}.png'))
    plt.close()

    # Save results to text file
    result_txt = os.path.join(output_dir, f'results_{seed}.txt')
    with open(result_txt, 'a') as f:
        f.write(f"Seed: {seed}\n")
        f.write(
            f"Accuracy: {acc:.4f}, F1: {f1:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}, AUC: {auc_score:.4f}\n")

        # Generate classification report
        classification_rep = classification_report(y_true, y_pred, target_names=EMOS, digits=4)
        f.write(f"Classification Report:\n{classification_rep}\n")
        f.write("\n")

    # Clear memory to prevent memory leaks
    tf.keras.backend.clear_session()

    # Return evaluation metrics
    return {'seed': seed, 'accuracy': acc, 'recall': recall, 'f1': f1, 'precision': precision, 'auc': auc_score}


# List to store each run's results
results = []

# Run 10 iterations, each with a different random seed
for seed in range(10):  # For example, loop 10 times
    result = train_and_evaluate(seed)
    results.append(result)

# Print results from each run
for result in results:
    print(
        f"Seed: {result['seed']}, Accuracy: {result['accuracy']:.4f}, F1: {result['f1']:.4f}, Recall: {result['recall']:.4f}, Precision: {result['precision']:.4f}, AUC: {result['auc']:.4f}")

# Find the random seed that produced the best results
best_result = max(results, key=lambda x: x['accuracy'])
print(f"\nBest seed: {best_result['seed']}, Best Accuracy: {best_result['accuracy']:.4f}")

# --- Comparison plots for all seeds ---
import matplotlib.pyplot as plt

seeds = [result['seed'] for result in results]
accuracies = [result['accuracy'] for result in results]
recalls = [result['recall'] for result in results]
f1s = [result['f1'] for result in results]
precisions = [result['precision'] for result in results]

# Accuracy comparison
plt.figure(figsize=(8, 5))
plt.plot(seeds, accuracies, marker='o', color='blue', label='Accuracy')
plt.title('Accuracy per Seed')
plt.xlabel('Seed')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.legend()
plt.grid(True)
os.makedirs(os.path.join(output_dir, 'accuracy_plots'), exist_ok=True)
plt.savefig(os.path.join(output_dir, 'accuracy_plots', 'accuracy_per_seed.png'))
plt.close()

# Recall comparison
plt.figure(figsize=(8, 5))
plt.plot(seeds, recalls, marker='o', color='orange', label='Recall')
plt.title('Recall per Seed')
plt.xlabel('Seed')
plt.ylabel('Recall')
plt.ylim(0, 1)
plt.legend()
plt.grid(True)
os.makedirs(os.path.join(output_dir, 'recall_plots'), exist_ok=True)
plt.savefig(os.path.join(output_dir, 'recall_plots', 'recall_per_seed.png'))
plt.close()

# F1 Score comparison
plt.figure(figsize=(8, 5))
plt.plot(seeds, f1s, marker='o', color='green', label='F1 Score')
plt.title('F1 Score per Seed')
plt.xlabel('Seed')
plt.ylabel('F1 Score')
plt.ylim(0, 1)
plt.legend()
plt.grid(True)
os.makedirs(os.path.join(output_dir, 'f1_score_plots'), exist_ok=True)
plt.savefig(os.path.join(output_dir, 'f1_score_plots', 'f1_per_seed.png'))
plt.close()

# Precision comparison
plt.figure(figsize=(8, 5))
plt.plot(seeds, precisions, marker='o', color='red', label='Precision')
plt.title('Precision per Seed')
plt.xlabel('Seed')
plt.ylabel('Precision')
plt.ylim(0, 1)
plt.legend()
plt.grid(True)
os.makedirs(os.path.join(output_dir, 'precision_plots'), exist_ok=True)
plt.savefig(os.path.join(output_dir, 'precision_plots', 'precision_per_seed.png'))
plt.close()