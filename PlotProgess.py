import torch
import matplotlib.pyplot as plt
import os

checkpoint_dir = './Checkpoints'
# Load the last saved checkpoint
file_path = os.path.join(checkpoint_dir, 'training_checkpoint.pth')
checkpoint = torch.load(file_path, map_location=torch.device('cpu'))
history = checkpoint['history']
# For a quick look at the raw data
print(history)

# For a cleaner, epoch-by-epoch view
print("--- Training History ---")
for epoch in range(len(history['train_loss'])):
    print(f"Epoch {epoch+1}: "
          f"Train Loss: {history['train_loss'][epoch]:.4f}, Train Acc: {history['train_acc'][epoch]:.4f} | "
          f"Val Loss: {history['val_loss'][epoch]:.4f}, Val Acc: {history['val_acc'][epoch]:.4f}")
# Create a plot with two subplots: one for accuracy, one for loss
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# --- Plot Accuracy ---
ax1.plot(history['train_acc'], label='Train Accuracy')
ax1.plot(history['val_acc'], label='Validation Accuracy')
ax1.set_title('Model Accuracy')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True)


# --- Plot Loss ---
ax2.plot(history['train_loss'], label='Train Loss')
ax2.plot(history['val_loss'], label='Validation Loss')
ax2.set_title('Model Loss')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True)

plt.suptitle('Training and Validation Metrics')
plt.show()
