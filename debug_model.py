import torch
import torchvision

print(f"--- Diagnosing Model Architecture ---")
print(f"PyTorch Version: {torch.__version__}")
print(f"Torchvision Version: {torchvision.__version__}")

try:
    # We instantiate the exact model you are using in your script
    model = torchvision.models.efficientnet_b3(weights=None)

    print("\n--- Model Architecture ---")
    print(model)
    print("\n--- Diagnosis ---")
    print("✅ Successfully created a standard EfficientNet-B3 classifier.")
    print("If you see this message, the problem is likely not in the model definition itself.")

except Exception as e:
    print(f"\n--- ERROR ---")
    print(f"Failed to create the model. Error: {e}")