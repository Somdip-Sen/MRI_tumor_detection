import os
from pathlib import Path
import torch
import torch.nn as nn
import torchvision.models as models
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io

# Import the necessary MONAI transforms
from monai.transforms import (
    Compose, LoadImamd, EnsureChannelFirstd, CenterSpatialCropd,
    NormalizeIntensityd, Resized, ToTensord, Lambdad
)

# It's assumed you have this helper function in a file named Extra_transform.py
from Extra_transform import to_single_channel

# --- 1. Define Application and Model Details ---
app = FastAPI(title="Brain Tumor Detection API", version="1.0")

# Define the location of your model and the class names
MODEL_PATH = Path("Checkpoints/brain_tumor_classifier_best.pth")
CLASS_NAMES = ['glioma', 'healthy', 'meningioma', 'pituitary']
DEVICE = torch.device("cpu") # Run inference on CPU for this simple API

# --- 2. Load The Trained Model ---
# This function will load and prepare your model
def load_model(model_path):
    # IMPORTANT: Ensure this matches the model you trained (EfficientNet-B0)
    model = models.efficientnet_b0(weights=None)

    # Adapt the model for 1-channel input and 4 classes (same as in your training script)
    model.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, len(CLASS_NAMES))

    # Load the saved weights
    # Use map_location to ensure it loads correctly on CPU
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval() # Set model to evaluation mode
    return model

# --- 3. Define the Preprocessing Pipeline ---
# This must be IDENTICAL to the validation/test transform from your training script
preprocess_transform = Compose([
    # Note: We use LoadImaged but will pass a NumPy array, not a path
    EnsureChannelFirstd(keys=["image"], strict_check=False),
    Lambdad(keys="image", func=to_single_channel),
    Resized(keys="image", spatial_size=(256, 256)),
    CenterSpatialCropd(keys="image", roi_size=(224, 224)),
    ToTensord(keys="image"),
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
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
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    # Convert PIL Image to NumPy array
    image_np = np.array(image)
    
    # Create a dictionary for MONAI transforms
    data_dict = {"image": image_np}

    # Apply the preprocessing transforms
    processed_data = preprocess_transform(data_dict)
    input_tensor = processed_data["image"].unsqueeze(0).to(DEVICE) # Add batch dimension

    # Run prediction
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted_class_idx = torch.max(probabilities, 1)

    predicted_class_name = CLASS_NAMES[predicted_class_idx.item()]
    confidence_score = confidence.item()

    return {
        "predicted_class": predicted_class_name,
        "confidence": f"{confidence_score:.4f}"
    }
