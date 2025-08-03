import splitfolders
from collections import Counter
import random
from PIL import Image
import cv2
import numpy as np
import os
import time

from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
import torch, torchvision as tv, torch.optim as optim, copy
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms.functional as F
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from pathlib import Path
from monai.data import CacheDataset
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, RandSpatialCropd, RandFlipd, RandAffined,
    Rand2DElasticd, RandGaussianNoised, RandBiasFieldd, RandHistogramShiftd, CenterSpatialCropd,
    RandZoomd, RandRotate90d, NormalizeIntensityd, Resized, CenterSpatialCropd,
    RandGaussianSmoothd, RandShiftIntensityd, RandCoarseDropoutd, ToTensord, Lambdad
)
from torch.utils.data import WeightedRandomSampler

# 1) Sample one image per class
data_dir = Path("data/train")
classes = sorted([d for d in data_dir.iterdir() if d.is_dir()])
sample_paths = []
for cls in classes:
    imgs = list(cls.glob("*.jpg"))
    if imgs:
        sample_paths.append(random.choice(imgs))
from Extra_transform import to_single_channel

# 2) Define baseline (no random aug) and full augmentation pipelines
baseline_tf = Compose([
    LoadImaged(keys="image"),
    EnsureChannelFirstd(keys="image", strict_check=False),
    Lambdad(keys="image", func=to_single_channel),
    Resized(keys="image", spatial_size=(256, 256)),
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
])
aug_tf = Compose([
    LoadImaged(keys="image"),
    EnsureChannelFirstd(keys="image", strict_check=False),
    Lambdad(keys="image", func=to_single_channel),
    RandSpatialCropd(keys="image", roi_size=(256, 256), random_size=False),
    RandFlipd(keys="image", prob=0.5, spatial_axis=1),
    RandRotate90d(keys="image", prob=0.5),
    Resized(keys="image", spatial_size=(256, 256)),
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
])

data_dir = "data/train"
dataset = datasets.ImageFolder(root=data_dir, transform=None)
labels = [label for _, label in dataset.imgs]
counts = Counter(labels)  # Counter({1: 1600, 3: 1405, 2: 1316, 0: 1296})
class_names = dataset.classes
print({class_names[k]: v for k, v in counts.items()})

train_tf = Compose([
    # 1) basic loading ‒ keep exactly as before
    LoadImaged(keys="image"),
    EnsureChannelFirstd(keys="image", strict_check=False),
    Lambdad(keys="image", func=to_single_channel),

    # 2) mild geometric noise ––––––––––––––––––––––––––––
    RandSpatialCropd(keys="image", roi_size=(384, 384), random_size=False),
    RandFlipd(keys="image", prob=0.5, spatial_axis=0),  # LR flip
    RandRotate90d(keys="image", prob=0.3, max_k=1),  # ±90°
    RandAffined(  # tiny tilt/shear
        keys="image", prob=0.15,
        rotate_range=0.035,
        shear_range=0.017
    ),

    # 3) intensity / noise tweaks –––––––––––––––––––––––
    RandBiasFieldd(keys="image", prob=0.15),
    RandGaussianNoised(keys="image", prob=0.2, std=0.01),
    RandHistogramShiftd(keys="image", prob=0.05, num_control_points=3),
    RandShiftIntensityd(keys="image", offsets=0.05, prob=0.15),
    RandGaussianSmoothd(keys="image", sigma_x=(0.5, 1.0), prob=0.1),
    Rand2DElasticd(keys="image", prob=0.1, spacing=(32, 32), magnitude_range=(2, 5)),

    # 4) mild zoom last (keeps holes small)
    RandZoomd(keys="image", prob=0.2, min_zoom=0.9, max_zoom=1.1),

    # 5) very small random hole (Cut-out)
    RandCoarseDropoutd(
        keys="image",
        holes=1,  # just one hole
        spatial_size=(16, 16),  # smaller block removed
        prob=0.05
    ),

    # 6) final standardisation ––––––––––––––––––––––––––
    CenterSpatialCropd(keys="image", roi_size=(240, 240)),
    Resized(keys="image", spatial_size=(256, 256), mode="nearest"),
    ToTensord(keys="image"),
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
])

val_test_tf = Compose([
    # EnsureChannelFirstd(keys="image", channel_dim=0),
    LoadImaged(keys="image"),
    EnsureChannelFirstd(keys="image", strict_check=False),
    Lambdad(keys="image", func=to_single_channel),
    CenterSpatialCropd(keys="image", roi_size=(224, 224)),
    # centre crop -- we can otherwise resize to 224x224 but most medical CV papers keep the classic resize-then-centre-crop protocol
    # bcz 1. imagenet model was pretrained on 224 × 224 images 2. cropping removes the border artefacts safely
    Resized(keys="image", spatial_size=(256, 256)),  # long edge → 256
    ToTensord(keys="image"),
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True)
    # hand-off to PyTorch tensor and ImageNet stats (1-channel version)
])


def sanity_check(loader, device="cpu", n_batches=2):
    '''loader pull two real batches with workers running, moves them to the GPU, and times
    the whole trip so you know that (a) the pipeline doesn’t crash once workers/GPU enter
    the picture and (b) num_workers choice is feeding the GPU fast enough.'''
    t0 = time.perf_counter()
    for b, batch in enumerate(loader):
        if b >= n_batches:
            break
        x = batch["image"]
        y = batch["label"]
        print(f"Batch {b}: tensor {x.shape} dtype={x.dtype} device={x.device}")
        assert x.shape[1:] == (1, 256, 256), "Wrong spatial dims or channels!"
        x = x.to(device)
        y = y.to(device)
    dt = time.perf_counter() - t0
    imgs_per_sec = (n_batches * loader.batch_size) / dt
    print(f"✓ {n_batches * loader.batch_size} images OK "
          f"@ {imgs_per_sec:,.0f} img/s")


def get_file_list_for_split(split_path):
    classes = sorted(d for d in split_path.iterdir() if d.is_dir())
    label_map = {cls.name: idx for idx, cls in enumerate(classes)}
    files = []
    for cls in classes:
        for f in cls.rglob("*.jpg"):
            files.append({
                "image": str(f),
                "label": label_map[cls.name]
            })

    # print(files[0:5])
    return files


if __name__ == '__main__':
    NUM_EPOCHS = 100
    PATIENCE = 5
    BATCH = 32
    NUM_W = 24
    NUM_WARMUP_EPOCHS = 3
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    ROOT = Path("data")
    train_path = ROOT / "train"
    val_path = ROOT / "val"
    test_path = ROOT / "test"

    train_files = get_file_list_for_split(train_path)
    val_files = get_file_list_for_split(val_path)
    test_files = get_file_list_for_split(test_path)

    train_ds = CacheDataset(train_files, transform=train_tf, num_workers=NUM_W, cache_rate=0.5)

    # Build the sampler so every batch is balanced
    target_ratios = {'glioma': 0.25, 'healthy': 0.25,
                     'meningioma': 0.25, 'pituitary': 0.25}
    label2idx = {'glioma': 0,
                 'healthy': 1,
                 'meningioma': 2,
                 'pituitary': 3}

    # label_list is already the integer label for every training image
    label_list = torch.tensor([f["label"] for f in train_files])

    # 1.  count how many images each label really has
    class_counts = torch.bincount(label_list, minlength=4).float()  # → tensor([1296,1600,1316,1405])

    # 2.  inverse-frequency per class
    inv_freq = 1.0 / class_counts

    # 3) per-sample weight list  (length = len(train_files))
    sample_weights = inv_freq[label_list]  # fancy-indexing

    sampler = WeightedRandomSampler(weights=sample_weights,
                                    num_samples=len(sample_weights),
                                    replacement=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH, num_workers=NUM_W, sampler=sampler, shuffle=False,
                              pin_memory=True)

    val_ds = CacheDataset(val_files, transform=val_test_tf, num_workers=NUM_W, cache_rate=0.2)

    val_loader = DataLoader(val_ds, batch_size=BATCH, num_workers=NUM_W, pin_memory=True)
    val_batch = next(iter(val_loader))

    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

    original_first_layer = model.features[0][0]

    new_first_layer = nn.Conv2d(
        in_channels=1,
        out_channels=original_first_layer.out_channels,
        kernel_size=original_first_layer.kernel_size,
        stride=original_first_layer.stride,
        padding=original_first_layer.padding,
        bias=(original_first_layer.bias is not None)
    )

    with torch.no_grad():
        new_first_layer.weight[:] = original_first_layer.weight.mean(dim=1, keepdim=True)

        model.features[0][0] = new_first_layer
        if original_first_layer.bias is not None:
            new_first_layer.bias.copy_(original_first_layer.bias)

    NUM_CLASSES = 4

    num_filters_before_final_layer_output = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(num_filters_before_final_layer_output, NUM_CLASSES)
    )

    for param in model.parameters():
        param.requires_grad = False

    for param in model.features[0][0].parameters():
        param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True

    for param in model.features[-1].parameters():
        param.requires_grad = True

    for param in model.features[-2].parameters():
        param.requires_grad = True

    model = model.to(device)
    model = torch.compile(model)
    print("Model successfully adapted for 1-channel input and 4-class output.")

    USE_FOCAL = False
    if USE_FOCAL:
        class FocalLoss(nn.Module):
            def __init__(self, gamma=2.0):
                super().__init__()
                self.gamma = gamma

            def forward(self, logits, target):
                p = torch.softmax(logits, 1)
                pt = p.gather(1, target.unsqueeze(1)).squeeze()
                return (-((1 - pt) ** self.gamma) * torch.log(pt)).mean()


        criterion = FocalLoss()
    else:
        cls_weights = torch.tensor([2.0, 1.0, 1.8, 2.2], device=device)
        criterion = nn.CrossEntropyLoss(weight=cls_weights)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=1e-4, weight_decay=1e-5)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=3, verbose=True)


    def run_epoch(model, dataloader, criterion, optimizer=None, scaler=None, device="cpu", grad_clip_value=None):
        is_train = optimizer is not None
        model.train() if is_train else model.eval()

        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        progress_bar = tqdm(dataloader, desc="Train" if is_train else "Validate", leave=False)

        with torch.set_grad_enabled(is_train):
            for batch in progress_bar:
                inputs, labels = batch["image"].to(device), batch["label"].to(device)

                with torch.amp.autocast(device_type=str(device), dtype=torch.float16):
                    logits = model(inputs)
                    loss = criterion(logits, labels)

                if is_train:
                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    if grad_clip_value:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_value)
                    scaler.step(optimizer)  # or optimizer.step() if you disable AMP
                    scaler.update()

                preds = logits.argmax(dim=1)
                total_loss += loss.item() * labels.size(0)
                correct_predictions += (preds == labels).sum().item()
                total_samples += labels.size(0)

                current_loss = total_loss / total_samples
                current_acc = correct_predictions / total_samples
                progress_bar.set_postfix(loss=f"{current_loss:.4f}", acc=f"{current_acc:.4f}")

        final_loss = total_loss / total_samples
        final_acc = correct_predictions / total_samples

        return final_loss, final_acc


    scaler = torch.cuda.amp.GradScaler()
    CHECKPOINT_DIR = "Checkpoints"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    CHECKPOINT_File_path = os.path.join(CHECKPOINT_DIR, "training_checkpoint.pth")

    start_epoch = 0
    best_val_acc = 0.0
    best_val_loss = float('inf')
    early_stop_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    if os.path.exists(CHECKPOINT_File_path):
        print(f"Checkpoint found! Resuming training from {CHECKPOINT_File_path}")
        checkpoint = torch.load(CHECKPOINT_File_path)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        start_epoch = checkpoint['epoch']
        best_val_acc = checkpoint['best_val_acc']
        history = checkpoint['history']

        print(f"-> Resuming from epoch {start_epoch} with best validation accuracy of {best_val_acc:.4f}")

    else:
        print("No checkpoint found. Starting training from scratch.")

    start_time = time.time()
    Flag = True
    for epoch in range(start_epoch, NUM_EPOCHS):
        epoch_start_time = time.time()

        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, scaler=scaler, device=device, grad_clip_value=1.0
        )

        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, optimizer=None, device=device
        )

        if Flag and epoch > NUM_WARMUP_EPOCHS:
            # ---- unfreeze all layers ---------------------------------------
            for p in model.parameters():
                p.requires_grad = True

            # rebuild optimiser with lower LR
            optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-5)

            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2,
                                          patience=3, verbose=True)

            # optional: re-instantiate GradScaler to reset internal state
            scaler = torch.cuda.amp.GradScaler()
            Flag = False
            # -------------------------------------------

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if epoch > NUM_WARMUP_EPOCHS:
            scheduler.step(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc

            print(f"\nEpoch {epoch + 1}: New best model found! Val Acc: {val_acc:.4f}. Saving checkpoint...")

            epoch_duration = time.time() - epoch_start_time
            print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
                  f"Duration: {epoch_duration / 60:.2f} minutes")

            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'history': history
            }

            BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "brain_tumor_classifier_best.pth")
            torch.save(checkpoint, CHECKPOINT_File_path)
            torch.save(model.state_dict(), BEST_MODEL_PATH)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            print(f"\nNo improvement in validation loss for {early_stop_counter} epoch(s).")

        if early_stop_counter >= PATIENCE:
            print(f"\nEarly stopping triggered after {PATIENCE} epochs of no improvement.")
            break

        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} | ...")

    print(f"\\nTraining finished.")

    training_duration = time.time() - start_time
    print(f"\n Training finished in {training_duration / 60:.2f} minutes.")
    print(f" \nBest Validation Accuracy: {best_val_acc:.4f}")
