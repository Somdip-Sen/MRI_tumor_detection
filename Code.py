from torchvision import datasets
import splitfolders
from collections import Counter
import matplotlib.pyplot as plt
import random
from PIL import Image
import cv2
import numpy as np
import os

# Split the Data/train folder into train, val, and test subfolders
# splitfolders.ratio(
#     input="./Data/data", # Input folder with class subfolders
#     output="./Data",      # Output folder where train/val/test folders will be created
#     seed=43,               # For reproducibility
#     ratio=(0.7, 0.15, 0.15), # 70% train, 15% val, 15% test
#     group_prefix=None      # Default: no grouping
##    move = True         # files will not copy rather move
# )

data_dir = "data/train"
dataset = datasets.ImageFolder(root=data_dir, transform=None)
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
# # visualize_samples(dataset, num_samples=5)
#
#
# ## Check Resolution and Size Consistency
resolutions = []
file_sizes = []

for path, _ in dataset.imgs:
    img = cv2.imread(path)
    h, w = img.shape[:2]
    resolutions.append((h, w))
    file_sizes.append(os.path.getsize(path))

res_arr = np.array(resolutions)
print("Resolution stats (h×w):", res_arr.min(axis=0), res_arr.max(axis=0), res_arr.mean(axis=0))

import matplotlib.pyplot as plt
plt.hist([h for h, w in resolutions], bins=30, alpha=0.5, label="Heights")
plt.hist([w for h, w in resolutions], bins=30, alpha=0.5, label="Widths")
plt.legend(); plt.title("Resolution Distribution")
plt.show()

plt.hist([fs / 1e3 for fs in file_sizes], bins=30)
plt.xlabel("File size (KB)"); plt.ylabel("Count")
plt.title("Image file size distribution")
plt.show()

