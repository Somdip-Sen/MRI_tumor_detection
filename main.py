import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io

# Import the necessary MONAI transforms
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, CenterSpatialCropd,
    NormalizeIntensityd, Resized, ToTensord, Lambdad
)

# It's assumed you have this helper function in a file named Extra_transform.py
from Extra_transform import to_single_channel

# --- 1. Define Application and Model Details ---
app = FastAPI(title="Brain Tumor Detection API", version="1.0")

# Define the location of your model and the class names
MODEL_PATH = Path("Checkpoints/brain_tumor_classifier_best.pth")
CLASS_NAMES = ['glioma', 'healthy', 'meningioma', 'pituitary']

# Run inference on CPU for this simple API, but we can use GPU if available
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


# --- 2. Load The Trained Model ---
# This function will load and prepare your model
def load_model(model_path):
    # IMPORTANT: Ensure this matches the model you trained (EfficientNet-B0)
    model = models.efficientnet_b0(weights=None)
    original_first_layer = model.features[0][0]  # initial layer modification

    # Adapt the model for 1-channel input and 4 classes (same as in your training script)
    model.features[0][0] = nn.Conv2d(in_channels=1,
                                     out_channels=original_first_layer.out_channels,
                                     kernel_size=original_first_layer.kernel_size,
                                     stride=original_first_layer.stride,
                                     padding=original_first_layer.padding,
                                     bias=(original_first_layer.bias is not None)
                                     )
    # Get the number of input features for the classifier
    num_features = model.classifier[1].in_features

    # Replace the classifier with a new one for our 4 classes
    model.classifier[1] = nn.Linear(num_features, len(CLASS_NAMES))



    # Load the saved weights
    model.to(DEVICE)

    # Compile the model FIRST. "reduce-overhead" is a good mode for inference.
    model = torch.compile(model, mode="reduce-overhead")

    # Use map_location to ensure it loads correctly on CPU/GPU
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))

    model.eval()  # Set model to evaluation mode
    return model


# --- 3. Define the Preprocessing Pipeline ---
preprocess_transform = Compose([
    # No loadImaged as it assumes the input is a file path (string).But we're feeding it a NumPy array
    EnsureChannelFirstd(keys="image", strict_check=False),
    ToTensord(keys="image"), # Convert to a Tensor FIRST bcz The line t.mean(dim=0, ...) inside your to_single_channel function fails
    # will fail because NumPy arrays do not understand the dim argument (they use axis instead), while PyTorch Tensors do
    # in training code.py LoadImaged, reads the image file from the disk and immediately converts it into a PyTorch Tensor
    Lambdad(keys="image", func=to_single_channel),
    CenterSpatialCropd(keys="image", roi_size=(224, 224)),
    # centre crop -- we can otherwise resize to 224x224 but most medical CV papers keep the classic resize-then-centre-crop protocol
    # bcz 1. imagenet model was pretrained on 224 × 224 images 2. cropping removes the border artefacts safely
    Resized(keys="image", spatial_size=(256, 256)),  # long edge → 256
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True)
])

# Load the model when the application starts
model = load_model(MODEL_PATH)


# --- 4. Define the API Endpoints ---
@app.get("/")
def home():
    """A simple health check endpoint."""
    return {"status": "ok", "message": "API is running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accepts an image file, preprocesses it, runs inference, and returns the prediction.
    """
    # Read the image file uploaded by the user
    image_bytes = await file.read()
    # Load image using PIL and convert to RGB
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

    # Convert PIL Image to NumPy array
    image_np = np.array(image)

    # Create a dictionary wrap for MONAI transforms support
    data_dict = {"image": image_np}

    # Apply the preprocessing transforms
    processed_data = preprocess_transform(data_dict)
    input_tensor = processed_data["image"].unsqueeze(0).to(DEVICE)  # .unsqueeze(0) adds a new batch
    # dimension at position 0, turning the image from: [1, 256, 256] → [1, 1, 256, 256]

    # Run prediction
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        print("probabilities:", probabilities) # probabilities: metatensor([[0.0099, 0.9343, 0.0508, 0.0050]], device='mps:0')
        confidence, predicted_class_idx = torch.max(probabilities, 1)

    predicted_class_name = CLASS_NAMES[predicted_class_idx.item()]
    confidence_score = confidence.item()

    return {
        "predicted_class": predicted_class_name,
        "confidence": f"{confidence_score:.4f}"
    }
