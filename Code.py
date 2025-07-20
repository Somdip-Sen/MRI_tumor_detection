import splitfolders
from collections import Counter
import matplotlib.pyplot as plt
import random
from PIL import Image
import cv2
import numpy as np
import os
import time

import torch, torchvision as tv
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms.functional as F

from pathlib import Path

from monai.data import CacheDataset
from monai.transforms import (
    Compose, EnsureChannelFirstd, RandSpatialCropd, RandFlipd, RandAffined,
    Rand2DElasticd, RandGaussianNoised, RandBiasFieldd, RandHistogramShiftd,
    RandZoomd, RandRotate90d, NormalizeIntensityd, Resized, CenterSpatialCropd,
    RandGaussianSmoothd, RandShiftIntensityd, RandCoarseDropoutd
)

# # Split the Data/train folder into train, val, and test subfolders
# splitfolders.ratio(
#     input="./Data/data", # Input folder with class subfolders
#     output="./Data",      # Output folder where train/val/test folders will be created
#     seed=43,               # For reproducibility
#     ratio=(0.8, 0.1, 0.1), # 80% train, 10% val, 10% test
#     group_prefix=None      # Default: no grouping
# #    move = True         # files will not copy rather move
# )


# ── Understanding data - EDA ──────────────────────────────────────────────
# def count_images_in_directory(base_path):
#     """Counts images in subfolders for train, val, and test splits."""
#     splits = ['train', 'val', 'test']
#     class_counts = {}
#     for split in splits:
#         split_path = os.path.join(base_path, split)
#         if not os.path.exists(split_path):
#             print(f"Warning: {split_path} does not exist. Skipping.")
#             continue
#
#         class_counts[split] = {}
#         for class_name in os.listdir(split_path):
#             class_path = os.path.join(split_path, class_name)
#             if os.path.isdir(class_path):
#                 num_images = len([f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))])
#                 class_counts[split][class_name] = num_images
#     return class_counts
#
# base_data_path = 'Data'
# image_counts = count_images_in_directory(base_data_path)
#
# print("\n--- Summary of Image Counts ---")
# for split, counts in image_counts.items():
#     total_split_images = sum(counts.values())
#     print(f"{split.upper()} Set Total: {total_split_images} images ----")
#     for class_name, count in counts.items():
#         print(f"  {class_name}: {count} images")


# # Function to load and visualize an image properties
# def inspect_image(image_path, label):
#     if not os.path.exists(image_path):
#         print(f"Error: Image not found at {image_path}")
#         return
#
#     img = cv2.imread(image_path)
#
#     if img is None:
#         print(f"Error: Could not load image from {image_path}. Check file integrity.")
#         return
#
#     print(f"\n--- Inspecting {label} Image: {os.path.basename(image_path)} ---")
#     print(f"Image Shape (Height, Width, Channels): {img.shape}")
#     print(f"Image Data Type: {img.dtype}")
#     # You can also print min/max pixel values to check range
#     print(f"Min pixel value: {np.min(img)}")
#     print(f"Max pixel value: {np.max(img)}")
#
# # Inspect a healthy image
# inspect_image("Data/train/healthy/0000.jpg", "Healthy") # (225, 225, 3)
#
# # Inspect a glioma image (representing a tumor)
# inspect_image("Data/train/glioma/0000.jpg", "Glioma (Tumor)") # (512, 512, 3)


# data_dir = "data/train"
# dataset = datasets.ImageFolder(root=data_dir, transform=None)
# # print("Classes found:", dataset.classes)
# # print("Class to index mapping:", dataset.class_to_idx)
# # print("Total images:", len(dataset))
# labels = [label for _, label in dataset.imgs]
# counts = Counter(labels)
# # print("Class label counts:", counts)
# class_names = dataset.classes
# # print({class_names[k]: v for k, v in counts.items()}) # {'glioma': 1471, 'healthy': 1823, 'meningioma': 1498, 'pituitary': 1601}
#
# ## Visualize Sample Images from Each Class
# def visualize_samples(dataset, num_samples=3):     # visualize random 3 sample from all the classes
#     fig, axes = plt.subplots(len(dataset.classes), num_samples, figsize=(num_samples * 3, len(dataset.classes) * 3))
#     for cls_idx, cls_name in enumerate(dataset.classes):
#         indices = [i for i, (_, label) in enumerate(dataset.imgs) if label == cls_idx]
#         selected = random.sample(indices, num_samples)
#         for j, idx in enumerate(selected):
#             path, _ = dataset.imgs[idx]
#             img = Image.open(path).convert("RGB")
#             axes[cls_idx, j].imshow(img)
#             axes[cls_idx, j].axis("off")
#             if j == 0:
#                 axes[cls_idx, j].set_title(cls_name)
#     plt.tight_layout()
#     plt.show()
#
# visualize_samples(dataset, num_samples=5)


# # Check Resolution and Size Consistency
# resolutions = []
# file_sizes = []
#
# for path, _ in dataset.imgs:
#     img = cv2.imread(path)
#     h, w = img.shape[:2]
#     resolutions.append((h, w))
#     file_sizes.append(os.path.getsize(path))
#
# res_arr = np.array(resolutions)
# print("Resolution stats (h×w):", res_arr.min(axis=0), res_arr.max(axis=0), res_arr.mean(axis=0))
#
# import matplotlib.pyplot as plt
# plt.hist([h for h, w in resolutions], bins=30, alpha=0.5, label="Heights")
# plt.hist([w for h, w in resolutions], bins=30, alpha=0.5, label="Widths")
# plt.legend(); plt.title("Resolution Distribution")
# plt.show()
#
# plt.hist([fs / 1e3 for fs in file_sizes], bins=30)
# plt.xlabel("File size (KB)"); plt.ylabel("Count")
# plt.title("Image file size distribution")
# plt.show()


# ── DATA Preprocessing ──────────────────────────────────────────────
'''
1. MONAI specific prebuilt data augmentation functions (for training only)
2. Standardizing all the images to 224x224 pixels and normalizing the utf-8 (8bit => 0-255) pixel values to 0-1
3. 

'''
ROOT = Path("Data")
BATCH = 4  #32
NUM_W = 4  # tune to your CPU cores
DEVICE = torch.device("mps")

# ── TRANSFORM PIPELINES ─────────────────────────────────────────────

## basic transformations ---
# train_tf = transforms.Compose([
#     transforms.Resize(256),                          # fix width variability
#     transforms.RandomResizedCrop(224, scale=(0.8,1)),# random zoom crop
#     transforms.RandomHorizontalFlip(p=0.5),
#     transforms.RandomRotation(degrees=15),
#     transforms.ColorJitter(brightness=.15,
#                            contrast=.15),
#     transforms.ToTensor(),
#     transforms.Normalize(mean, std),
# ])
# val_tf = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean, std),
# ])
## ---

## MONAI standard transformations ---
train_tf = Compose([
    EnsureChannelFirstd(keys="image"),
    # MONAI uses CHW. So Converting [H × W × C] → [C × H × W] for PyTorch kernels so PyTorch sees “channel first” => CHW order expected by EfficientNet.
    RandSpatialCropd(keys="image", roi_size=(256, 256), random_size=False),
    # Crops a random 256 × 256 patch (no resize) => gives position variety while preserving tumour context.
    RandFlipd(keys="image", prob=0.5, spatial_axis=1),
    # 50% chance left ↔ right flip along x-axis => Brain is roughly L/R symmetric; doubles data without clinical distortion.
    RandRotate90d(keys="image", prob=0.5, max_k=3),
    # random 0/90/180/270° rotation => Simulates different head orientations from scanners and PACS viewers
    RandAffined(keys="image", prob=0.3, rotate_range=0.08, shear_range=0.04),
    # Small random rotation (±4.5°), shear, translate. Mimics patient's head tilt & minor slice angulation; encourages geometric robustness
    Rand2DElasticd(keys="image", prob=0.2, spacing=(32, 32), magnitude_range=(2, 5)),
    # Non-linear elastic warp/Rubber-sheet warp (jelly-like).(magnitude 2–5 px) => Models patient motion / soft-tissue deformation / interpolation artefacts seen in multi-vendor scans.
    RandBiasFieldd(keys="image", prob=0.3),
    # Applies smooth multiplicative bias field => Replicates RF-coil inhomogeneity—common MRI artefact that shifts intensity across slice.
    # [ie Imagine photographing a page under a desk lamp: the centre is brighter, corners dimmer.  Bias-field augmentation teaches the model that the “letter shapes” (tumours) matter, not the lamp gradient.]
    RandGaussianNoised(keys="image", prob=0.2, std=0.01),
    # Adds Gaussian noise σ = 0.01 => Matches grain produced by low-SNR sequences or accelerated scans.
    RandHistogramShiftd(keys="image", prob=0.3, num_control_points=5),
    # Randomly bend the grey-level curve ie. move grey-level histogram control points => Simulates different window/level presets and vendor contrast curves.
    RandShiftIntensityd(keys="image", offsets=0.1, prob=0.3),
    # Uniform ±0.1 shift of all voxel intensities (brightness tweak) => Covers global brightness drift (scanner calibration).
    RandCoarseDropoutd(keys="image", holes=1, max_holes=3,  # Cuts out 1-3 random 32 × 32 squares
                       spatial_size=(32, 32), prob=0.15),
    # “Cut-Mix” style occlusion forces model to use wider context, not one hotspot.
    RandGaussianSmoothd(keys="image", sigma_x=(0.5, 1.5), prob=0.1),
    # Random blur (σ 0.5–1.5 px) => Imitates slice-thickness blur or patient motion; makes model blur-tolerant.
    RandZoomd(keys="image", prob=0.2, min_zoom=0.9, max_zoom=1.2),
    # Random zoom 0.9–1.2x => Handles FOV changes between scanners or axial vs oblique reconstructions.
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    # Z-score per channel (brain mask) => Removes scanner-specific intensity scale so network learns anatomy, not brightness.
    # ie Turns raw intensities into zero-mean, unit-std per slice
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(mean=(0.485,), std=(0.229,))
    # ImageNet mean/std ensure compatibility with pretrained EfficientNet ie mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225) but as we are dealing with grayscale so only 1 channel is given
])
# the 'd' in every transformation name signifies that the transformation operates on a dictionary of data instead of a bare Tensor like {"image": Tensor, "label": Tensor, …}

val_test_tf = Compose([
    EnsureChannelFirstd(keys="image"),
    Resized(keys="image", spatial_size=(256, 256)),  # long edge → 256
    CenterSpatialCropd(keys="image", roi_size=(224, 224)),
    # centre crop -- we can otherwise resize to 224x224 but most medical CV papers keep the classic resize-then-centre-crop protocol
    # bcz 1. imagenet model was pretrained on 224 × 224 images 2. cropping removes the border artefacts safely
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    # hand-off to PyTorch tensor and ImageNet stats (1-channel version)
    tv.transforms.ToTensor(),  # PIL → Tensor
    tv.transforms.Normalize(mean=(0.485,), std=(0.229,))
])


def sanity_check(loader, device="mps", n_batches=2):
    t0 = time.perf_counter()
    for b, (x, y) in enumerate(loader):
        if b >= n_batches: break  # don’t run full epoch
        print(f"Batch {b}: tensor {x.shape} dtype={x.dtype} device={x.device}")
        assert x.shape[1:] == (1, 224, 224), "Wrong spatial dims or channels!"
        # push to GPU once to confirm MPS works
        x = x.to(device);
        y = y.to(device)
    dt = time.perf_counter() - t0
    imgs_per_sec = (n_batches * loader.batch_size) / dt
    print(f"✓ {n_batches * loader.batch_size} images OK "
          f"@ {imgs_per_sec:,.0f} img/s")


def show_augmented_samples(loader, n=4):
    x, y = next(iter(loader))  # fresh batch (random augments)
    fig, ax = plt.subplots(1, n, figsize=(3 * n, 3))
    for i in range(n):
        img = F.to_pil_image(x[i])  # tensor ➜ PIL for display
        ax[i].imshow(img, cmap="gray")
        ax[i].set_title(f"label={y[i].item()}")
        ax[i].axis("off")
    plt.show()


## ---
if __name__ == "__main__":
    # ── DATASETS & LOADERS ──────────────────────────────────────────────
    # ********** this dataloader can't be used with MONAI because ImageFolder feeds a PIL Image,
    # but every transform in the list ends with d, which tells MONAI “I’m a dictionary transform—expect
    # a Python dict like {"image": image, "label": …}.". *****************
    #
    # def make_loader(split, tf, shuffle):
    #     ds = tv.datasets.ImageFolder(ROOT / split, transform=tf)
    #     return DataLoader(ds, batch_size=BATCH,
    #                       shuffle=shuffle,
    #                       num_workers=NUM_W,
    #                       pin_memory=False)
    #
    #
    # train_loader = make_loader("train", train_tf, True)
    # val_loader = make_loader("val", val_test_tf, False)
    # test_loader = make_loader("test", val_test_tf, False)
    #
    # print(f"Train imgs: {len(train_loader.dataset)}  "  # type: ignore[call-arg]
    #       f"Val imgs: {len(val_loader.dataset)}  "  # type: ignore[call-arg]
    #       f"Test imgs: {len(test_loader.dataset)}")  # type: ignore[call-arg]
    #***************
    # ── QUICK SANITY CHECK ──────────────────────────────────────────────
    # Fetch one mini-batch and push it through MPS to verify shapes/speeds.
    # x, y = next(iter(train_loader))
    # print("Batch:", x.shape, y.shape, "→ device:", DEVICE)
    # x = x.to(DEVICE);
    # y = y.to(DEVICE)

    files = [{"image": f} for f in (ROOT / "train").rglob("*.jpg")]  # dict -> {"image": Tensor, …}
    # print(*files[0:5])
    train_ds = CacheDataset(files, transform=train_tf, num_workers=0)
    # Compose some steps before the first Randomizable transform (e.g. LoadImage → EnsureChannelFirst → NormalizeIntensity) are executed and cached happens once (1st time and takes time)
    train_loader = DataLoader(train_ds, batch_size=32, num_workers=4)
    # takes the cached tensors in "train_ds" every batch and applies the random transforms that come after the cache-point (RandFlip, RandBiasField, etc.) and stacks size of BATCH = 32 tensors into a batch
    # If you skip CacheDataset, the whole Compose chain (deterministic + random) runs for every epoch.

    # ── QUICK SANITY CHECK ──────────────────────────────────────────────
    sanity_check(train_loader)  # prints shape / speed
    show_augmented_samples(train_loader)  # visual eyeball test

# Only deterministic transforms are in the val/test pipeline, so caching changes nothing,


# # --- MODEL DEFINITION ──────────────────────────────────────────────
# # Load a pre-trained EfficientNet-B0
# model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
#
# # Freeze all the parameters in the feature extraction layers
# for param in model.parameters():
#     param.requires_grad = False

# scaler = torch.amp.GradScaler(enabled=True)
# with torch.amp.autocast(device_type="mps"):
#     preds = model(inputs)
#     loss  = criterion(preds, labels)
# scaler.scale(loss).backward()
# scaler.step(optim); scaler.update()
# weights = torch.tensor([1/1471, 1/1823, 1/1498, 1/1601])[labels]
# sampler = WeightedRandomSampler(weights, num_samples=len(weights))
# train_loader = DataLoader(ds, batch_size=32,
#                           sampler=sampler, num_workers=4)
