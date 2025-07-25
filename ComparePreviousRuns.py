import torch
import matplotlib.pyplot as plt
import glob
import re
import os


def plot_all_runs(checkpoint_dir='./Checkpoints'):
    """
    Finds all 'training_checkpoint*.pth' files in a directory,
    loads their history, and plots a comparison graph.
    """
    # Use glob to find all files matching the pattern
    file_pattern = os.path.join(checkpoint_dir, 'training_checkpoint*.pth')
    checkpoint_files = sorted(glob.glob(file_pattern))  # Sorted for consistent colors

    if not checkpoint_files:
        print(f"No checkpoint files found in '{checkpoint_dir}' with pattern 'training_checkpoint*.pth'")
        return

    print(f"Found {len(checkpoint_files)} checkpoint(s) to compare: {checkpoint_files}")

    # Create the plot with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Loop through each found checkpoint file
    for file_path in checkpoint_files:

        # --- Extract a clean name for the legend ---
        filename = os.path.basename(file_path)
        match = re.search(r'(\d+)', filename)
        if match:
            # Found a number, like in 'training_checkpoint1.pth'
            exp_name = f"Run {match.group(1)}"
        else:
            # No number, so it's probably the base 'training_checkpoint.pth'
            exp_name = "Base Run"

        # --- Load the history from the checkpoint ---
        try:
            checkpoint = torch.load(file_path)
            if 'history' not in checkpoint:
                print(f"--> Skipping {filename}: 'history' key not found.")
                continue
            history = checkpoint['history']
        except Exception as e:
            print(f"--> Skipping {filename}: Could not load file. Error: {e}")
            continue

        # --- Plot metrics for the current run ---
        # Using solid lines for training and dashed lines for validation
        epochs = range(len(history['train_acc']))

        # Plot Accuracy
        ax1.plot(epochs, history['train_acc'], linestyle='-', label=f'Train Acc ({exp_name})')
        ax1.plot(epochs, history['val_acc'], linestyle='--', marker='o', markersize=4, label=f'Val Acc ({exp_name})')

        # Plot Loss
        ax2.plot(epochs, history['train_loss'], linestyle='-', label=f'Train Loss ({exp_name})')
        ax2.plot(epochs, history['val_loss'], linestyle='--', marker='o', markersize=4, label=f'Val Loss ({exp_name})')

    # --- Formatting the final plots ---
    ax1.set_title('Model Accuracy Comparison', fontsize=16)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)

    ax2.set_title('Model Loss Comparison', fontsize=16)
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

    plt.suptitle('Comparison of All Training Runs', fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for suptitle
    plt.show()


if __name__ == '__main__':
    plot_all_runs()