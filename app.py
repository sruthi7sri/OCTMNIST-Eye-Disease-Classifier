import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# -------------------------------
# 1. Define the ImprovedCNN Model
# -------------------------------
class ImprovedCNN(nn.Module):
    def __init__(self):
        super(ImprovedCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 4)  # 4 classes as in your training

    def forward(self, x):
        # Expect input x of shape (batch, 1, 28, 28)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 28x28 -> 14x14
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 14x14 -> 7x7
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.fc2(x)
        return x

# -------------------------------------
# 2. Load the Saved Model (cached)
# -------------------------------------
@st.cache(allow_output_mutation=True)
def load_model():
    model = ImprovedCNN()
    # Load model weights (using CPU here; adjust map_location if needed)
    model.load_state_dict(torch.load("best_model_final.pth", map_location=torch.device('cpu')))
    model.eval()  # set model to evaluation mode
    return model

model = load_model()

# -------------------------------------
# 3. Define the Image Preprocessing
# -------------------------------------
# This transform matches what you did during training:
# - Convert image to grayscale (1 channel)
# - Resize to 28x28
# - Convert to tensor (which scales pixel values to [0, 1])
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Ensure image is 1-channel
    transforms.Resize((28, 28)),                  # Resize to 28x28
    transforms.ToTensor()
])

# -------------------------------------
# 4. Build the Streamlit User Interface
# -------------------------------------
st.title("OCTMNIST Classifier")
st.write("Upload an image (jpg, jpeg, or png) to predict its class.")

uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Open the image and display it
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Preprocess the image using the defined transform.
        # Convert image to grayscale if it isn't already.
        img_transformed = transform(image)
        # Add a batch dimension: shape becomes (1, 1, 28, 28)
        img_transformed = img_transformed.unsqueeze(0)
    except Exception as e:
        st.error(f"Error processing image: {e}")
    else:
        # Run inference using the model.
        with torch.no_grad():
            outputs = model(img_transformed)
            # Apply softmax to get class probabilities
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
        
        st.write(f"**Predicted Class:** {predicted_class}")
        st.write("**Class Probabilities:**")
        st.write(probabilities.squeeze().tolist())
