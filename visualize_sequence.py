# in cmd -> python visualize_single_sequence.py "Data directory" --save_dir "Plot directory"
import argparse
import os
import random
from pathlib import Path
import matplotlib.pyplot as plt

# Import all the necessary MONAI transforms from your pipeline
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ToTensord, Lambdad,
    RandSpatialCropd, RandFlipd, RandRotate90d, RandAffined,
    Rand2DElasticd, RandBiasFieldd, RandGaussianNoised,
    RandHistogramShiftd, RandShiftIntensityd, RandCoarseDropoutd,
    RandGaussianSmoothd, RandZoomd, CenterSpatialCropd, Resized
)

# It's assumed you have this helper function in a file named Extra_transform.py
from Extra_transform import to_single_channel


def visualize_full_sequence(root_dir: str, num_samples: int = 5, save_dir: str = None):
    """
    Loads random samples and visualizes the cumulative effect of a single,
    unbroken sequence of data augmentations.
    """
    train_dir = Path(root_dir) / 'train'
    if not train_dir.exists():
        print(f"Error: 'train' directory not found in '{root_dir}'")
        return

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        print(f"Plots will be saved to: {os.path.abspath(save_dir)}")

    class_dirs = sorted([d for d in train_dir.iterdir() if d.is_dir()])

    # --- 1. Define the full, ordered sequence of augmentations based on your train_tf ---
    # NOTE: Probabilities are set to 1.0 for consistent visualization.
    augmentations_in_sequence = [
        # 1) basic geometric prep
        ('SpatialCrop', RandSpatialCropd(keys="image", roi_size=(384, 384), random_size=False)),
        ('Flip', RandFlipd(keys="image", prob=0.5, spatial_axis=0)),
        ('Rotate90', RandRotate90d(keys="image", prob=0.3, max_k=1)),  # ±90° only
        ('Affine', RandAffined(keys="image", prob=0.15, rotate_range=0.035, shear_range=0.017)),

        # 2) small cut-out
        ('Dropout', RandCoarseDropoutd(keys="image", holes=1, spatial_size=(16, 16), prob=0.10)),

        # 3) intensity / noise variations
        ('BiasField', RandBiasFieldd(keys="image", prob=0.15)),
        ('GaussNoise', RandGaussianNoised(keys="image", prob=0.20, std=0.01)),
        ('HistoShift', RandHistogramShiftd(keys="image", prob=0.10, num_control_points=3)),
        ('IntensityShift', RandShiftIntensityd(keys="image", offsets=0.05, prob=0.20)),
        ('Smooth', RandGaussianSmoothd(keys="image", sigma_x=(0.5, 1.0), prob=0.10)),

        # 4) mild zoom last (keeps dropout blocks small)
        ('Zoom', RandZoomd(keys="image", prob=0.20, min_zoom=0.9, max_zoom=1.1)),

        # 5) final standardisation
        ('CenterCrop', CenterSpatialCropd(keys="image", roi_size=(240, 240))),
        ('Resized', Resized(keys="image", spatial_size=(256, 256), mode="nearest")),
    ]

    # --- 2. Create the base pipeline for loading ---
    base_transforms = [
        LoadImaged(keys="image"),
        EnsureChannelFirstd(keys="image", strict_check=False),
        Lambdad(keys="image", func=to_single_channel),
        ToTensord(keys="image"),
    ]

    # --- 3. Programmatically build the cumulative pipelines ---
    pipelines = {"Original": Compose(base_transforms)}
    current_augs = list(base_transforms)
    for name, aug in augmentations_in_sequence:
        current_augs.append(aug)
        pipelines[f"+ {name}"] = Compose(current_augs)

    # --- 4. Loop through each class and visualize ---
    for class_dir in class_dirs:
        try:
            image_files = list(class_dir.glob("*.jpg"))
            sample_paths = random.sample(image_files, num_samples)
        except Exception as e:
            print(f"Could not read images from {class_dir}: {e}")
            continue

        num_steps = len(pipelines)
        # Increase figsize width to accommodate all steps
        fig, axes = plt.subplots(num_samples, num_steps, figsize=(num_steps * 3, num_samples * 3))
        fig.suptitle(f"Full Augmentation Sequence for Class: '{class_dir.name}'", fontsize=20)

        for row, img_path in enumerate(sample_paths):
            image_dict = {"image": str(img_path)}

            for col, (name, pipeline) in enumerate(pipelines.items()):
                augmented_dict = pipeline(image_dict)
                img_tensor = augmented_dict['image']
                img_to_show = img_tensor.squeeze().cpu().numpy()

                ax = axes[row, col] if num_samples > 1 else axes[col]
                ax.imshow(img_to_show, cmap='gray')
                ax.axis('off')

                if row == 0:
                    ax.set_title(name, fontsize=10)  # Smaller font for more titles

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        if save_dir:
            # save_path = os.path.join(save_dir, f"{class_dir.name}_sequence.png")
            save_path = os.path.join(save_dir, f"{class_dir.name}_sequence.svg")
            # save_path = os.path.join(save_dir, f"{class_dir.name}_sequence.pdf")
            print(f"Saving plot to {save_path}...")
            plt.savefig(save_path, bbox_inches='tight') # if problem dial back to 300-600 or dial up if needed

        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Visualize a single, full sequence of MONAI data augmentations."
    )
    parser.add_argument(
        "root_dir",
        type=str,
        help="The root directory of your dataset (e.g., './Data')."
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Optional. Directory to save the output plots as high-res images."
    )
    args = parser.parse_args()

    visualize_full_sequence(args.root_dir, num_samples=5, save_dir=args.save_dir)