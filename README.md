# MRI_tumor_detection
Build a PyTorch EfficientNet-B0 model to sort 6.4 k brain MRI slices into glioma, meningioma, pituitary or healthy. Data split 80-10-10, resized 224², augmented, trained with Adam+focal loss, ≥95 % acc. Grad-CAM heat-maps and a Streamlit demo plus Docker+FastAPI cloud API complete the end-to-end pipeline.
