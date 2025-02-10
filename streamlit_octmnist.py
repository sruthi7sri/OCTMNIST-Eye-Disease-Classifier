import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image

# Function to load model dynamically

def load_model(model_path):
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model = nn.Sequential(*list(checkpoint.keys()))  # Auto-load layers
    model.load_state_dict(checkpoint)
    model.eval()
    return model

# Load the model
model = load_model("best_model_final.pth")

# Define preprocessing function
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    image = transform(image).unsqueeze(0)
    return image

# Apply CLAHE
def apply_CLAHE(image_np, clip_limit=1.5, grid_size=10):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    equalized = clahe.apply((image_np * 255).astype(np.uint8))
    return equalized.astype(np.float32) / 255.0

# Streamlit UI
st.title("OCTMNIST Image Classification")
st.write("Upload an OCT image to classify it using the trained model.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess image
    image_np = np.array(image)
    clahe_image = apply_CLAHE(image_np)
    image_preprocessed = preprocess_image(Image.fromarray(clahe_image))
    
    # Make prediction
    with torch.no_grad():
        output = model(image_preprocessed)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
    
    # Class mapping
    class_labels = {0: "Class 0", 1: "Class 1", 2: "Class 2", 3: "Class 3"}
    
    st.write(f"### Prediction: {class_labels[predicted_class]}")
    st.write("### Confidence Scores:")
    for i, score in enumerate(probabilities.squeeze().tolist()):
        st.write(f"{class_labels[i]}: {score * 100:.2f}%")
