import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from monai.data import CacheDataset
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, CenterSpatialCropd,
    NormalizeIntensityd, Resized, ToTensord, Lambdad
)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# It's assumed you have this helper function in a file named Extra_transform.py
from Extra_transform import to_single_channel


def evaluate_model(model_path, data_dir, batch_size):
    """
    Loads a trained model and evaluates its performance on the test set.
    """
    # --- Setup ---
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    print(f"Using device: {device}")

    # --- Load Test Data ---
    test_path = Path(data_dir) / 'test'
    class_names = sorted([d.name for d in test_path.iterdir() if d.is_dir()])
    print(f"Found classes: {class_names}")

    def get_file_list(split_path):
        classes = sorted([d for d in split_path.iterdir() if d.is_dir()])
        label_map = {cls.name: i for i, cls in enumerate(classes)}
        file_list = []
        for cls in classes:
            for f in cls.rglob("*.jpg"):
                file_list.append({"image": str(f), "label": label_map[cls.name]})
        return file_list

    test_files = get_file_list(test_path)
    print(f"Found {len(test_files)} files in the test set.")

    # Use the same validation transforms for testing
    test_tf = Compose([
        LoadImaged(keys="image"),
        EnsureChannelFirstd(keys="image", strict_check=False),
        Lambdad(keys="image", func=to_single_channel),
        Resized(keys="image", spatial_size=(256, 256)),
        CenterSpatialCropd(keys="image", roi_size=(224, 224)),
        ToTensord(keys="image"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ])

    test_ds = CacheDataset(test_files, transform=test_tf, cache_rate=0.1)
    test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=8, pin_memory=True)

    # --- Load Model ---
    # IMPORTANT: Ensure this matches the model you trained (e.g., efficientnet_b3)
    model = models.efficientnet_b3(weights=None)

    # Adapt the model for 1-channel input and 4 classes
    original_first_layer = model.features[0][0]
    new_first_layer = nn.Conv2d(1, original_first_layer.out_channels, kernel_size=original_first_layer.kernel_size,
                                stride=original_first_layer.stride, padding=original_first_layer.padding, bias=False)
    model.features[0][0] = new_first_layer

    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(num_features, len(class_names))
    )

    # Load the saved weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model = torch.compile(model)  # Compile for faster inference
    model.eval()

    # --- Run Inference ---
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating on Test Set"):
            inputs, labels = batch["image"].to(device), batch["label"].to(device)

            with torch.amp.autocast(device_type=str(device), dtype=torch.float16):
                logits = model(inputs)

            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # --- Calculate and Display Metrics ---
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"\n--- Test Set Evaluation ---")
    print(f"Overall Test Accuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    print("Confusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix for Test Set')

    # On a server, you'd save the plot; on a local machine, you can show it.
    plt.savefig('confusion_matrix.png')
    print("\nConfusion matrix plot saved to 'confusion_matrix.png'")
    # plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a trained brain tumor classification model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved .pth model weights file.")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root directory of the dataset (containing the 'test' folder).")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for evaluation.")
    args = parser.parse_args()

    evaluate_model(args.model_path, args.data_dir, args.batch_size)