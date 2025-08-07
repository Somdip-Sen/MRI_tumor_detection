import io, os, logging
from pathlib import Path
from typing import Dict
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image

from contextlib import asynccontextmanager

# Import the necessary MONAI transforms for inference
from monai.transforms import (
    Compose, EnsureChannelFirstd, CenterSpatialCropd,
    NormalizeIntensityd, Resized, ToTensord, Lambdad
)
from Extra_transform import to_single_channel  # that helper function


# --- 2. Load The Trained Model ---
def load_model():
    """
    Model loader: prefer TorchScript .pt, else state_dict .pth
    This function will load and prepare my model
    :return: model
    """
    global BACKEND
    # A) Try TorchScript
    if TS_PATH.exists():
        try:
            logger.info(f"Loading TorchScript: {TS_PATH}")
            ts = torch.jit.load(str(TS_PATH), map_location=DEVICE)
            ts.eval()
            BACKEND = "torchscript"
            return ts
        except Exception as e:
            logger.warning(f"Failed to load TorchScript on {DEVICE}: {e}. Falling back to state_dict.")
    # B) Fallback: state_dict
    assert PTH_PATH.exists(), f"Missing model file: {PTH_PATH}"
    logger.info(f"Loading state_dict: {PTH_PATH}")
    model = build_efficientnet_b0_modified(len(CLASS_NAMES)).to(DEVICE)
    sd = torch.load(str(PTH_PATH), map_location=DEVICE)
    model.load_state_dict(sd)
    model.eval()
    # Optional compile on GPU only (skip on CPU)
    if DEVICE.type != "cpu":
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except Exception as e:
            logger.warning(f"torch.compile not available: {e}")
    BACKEND = "state_dict"
    return model


# --- 1. Define Application and Model Details ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup phase ---
    try:
        dummy = torch.zeros(1, 1, 256, 256, device=DEVICE)  # (batch=1, channels=1, height=256, width=256)
        with torch.no_grad():
            _ = model(dummy)
        logger.info(f"Warm-up done on {DEVICE} using backend={BACKEND}")
    except Exception as e:
        logger.warning(f"Warm-up skipped: {e}")

    # Yield control to the app (runs while API is live)
    yield

    # --- Shutdown phase (optional) ---
    # Place any cleanup code here if needed
    # e.g., closing database connections, releasing GPU memory, etc.


# API logging
app = FastAPI(title="Brain Tumor Detection API", version="1.1", lifespan=lifespan)
# print() won’t appear in some deployment logs unless redirected
logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
logger = logging.getLogger("api")

CLASS_NAMES = ['glioma', 'healthy', 'meningioma', 'pituitary']
CKPT_DIR = Path("Checkpoints")
TS_PATH = CKPT_DIR / "brain_tumor_classifier_best.pt"  # TorchScript (future optional)
PTH_PATH = CKPT_DIR / "brain_tumor_classifier_best.pth"  # state_dict (my current)

# Run inference on CPU for this simple API, but we can use GPU if available
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

BACKEND = "unknown"  # will be set to "torch script" or "state_dict"

preprocess_transform = Compose([
    EnsureChannelFirstd(keys="image", strict_check=False),
    ToTensord(keys="image"),
    Lambdad(keys="image", func=to_single_channel),
    CenterSpatialCropd(keys="image", roi_size=(224, 224)),
    Resized(keys="image", spatial_size=(256, 256)),
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
])
model = load_model()


def preprocess_transform_image(image_np: np.ndarray) -> torch.Tensor:
    """
    Apply the preprocessing transforms
    :param image_np:
    :return:Torch Tensor
    """
    data_dict = {"image": image_np}
    processed_data = preprocess_transform(data_dict)
    return processed_data["image"].unsqueeze(0)  # .unsqueeze(0) adds a new batch
    # dimension at position 0, turning the image from: [1, 256, 256] → [1, 1, 256, 256]


def build_efficientnet_b0_modified(num_classes: int = 4) -> nn.Module:
    """
    This function will load and prepare my model
    :param num_classes:
    :return: custom neural network model
    """
    model = models.efficientnet_b0(weights=None)
    original_first_layer = model.features[0][0]  # initial layer modification
    # Adapt the model for 1-channel input and 4 classes (same as in modely training script)
    model.features[0][0] = nn.Conv2d(
        in_channels=1,
        out_channels=original_first_layer.out_channels,
        kernel_size=original_first_layer.kernel_size,
        stride=original_first_layer.stride,
        padding=original_first_layer.padding,
        bias=(original_first_layer.bias is not None),
    )
    # Get the number of input features for the classifier
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, num_classes)
    return model


# Endpoints
@app.get("/")
def root():
    return {"status": "ok", "message": "API is running", "backend": BACKEND, "device": str(DEVICE)}


@app.get("/health")
def health():
    return {"status": "healthy"}


ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_UPLOAD_MB = 10  # <10MB file allowed
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> Dict:
    # MIME guard
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported content-type: {file.content_type}. "
                   f"Allowed: {sorted(ALLOWED_CONTENT_TYPES)}"
        )
    # Load & validate image
    try:
        image_bytes = await file.read()

        # size guard
        # if len(image_bytes) > MAX_UPLOAD_BYTES: --> peeking to the content header can do fast fail-check
        content_length = file.headers.get("content-length")
        if content_length and int(content_length) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File too large (> {MAX_UPLOAD_BYTES // (1024 * 1024)} MB)"
            )
        # Validate it’s a real image
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        # So that, If a client sends a non-image or corrupt file, FastAPI will not crash.
    except Exception as e:
        # classic FastAPI shape → {"detail": "..."}
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    # Preprocess
    image_np = np.array(img)
    input_tensor = preprocess_transform_image(image_np).to(DEVICE)

    # Inference
    with torch.inference_mode():  # instead of torch.no_grad() for a tiny speed bump
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)[0]

    top_idx = int(torch.argmax(probs).item())
    top_label = CLASS_NAMES[top_idx]
    top_conf = float(probs[top_idx].item())
    all_probs = {CLASS_NAMES[i]: float(probs[i].item()) for i in range(len(CLASS_NAMES))}
    logger.info(f"Probabilities: {all_probs}")

    return {
        "predicted_class": top_label,
        "confidence": round(top_conf, 4),
        "all_class_probabilities": all_probs,
        "backend": BACKEND,
        "device": str(DEVICE),
    }
