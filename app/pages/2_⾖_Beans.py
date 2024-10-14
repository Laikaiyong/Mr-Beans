import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import os

cwd = os.path.dirname(__file__) 

# Original CNN Model used by the app
class OriginalCNNModel(nn.Module):
    def __init__(self):
        super(OriginalCNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 14 * 14, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 4)  # Adjust for number of classes

    def forward(self, x):
        x = self.pool(nn.ReLU()(self.conv1(x)))
        x = self.pool(nn.ReLU()(self.conv2(x)))
        x = self.pool(nn.ReLU()(self.conv3(x)))
        x = self.pool2(x)
        x = x.view(-1, 128 * 14 * 14)
        x = nn.ReLU()(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Load the original CNN model
original_cnn_model_path = cwd + "/models/roasted_cnn_model.pth"
original_cnn_model = OriginalCNNModel()
original_cnn_model.load_state_dict(torch.load(original_cnn_model_path, map_location=torch.device('cpu')), strict = False)
original_cnn_model.eval()

# New CNN Model (trained)
class NewCNNModel(nn.Module):
    def __init__(self):
        super(NewCNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 14 * 14, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 4)  # Adjust for number of classes

    def forward(self, x):
        x = self.pool(nn.ReLU()(self.conv1(x)))
        x = self.pool(nn.ReLU()(self.conv2(x)))
        x = self.pool(nn.ReLU()(self.conv3(x)))
        x = self.pool2(x)
        x = x.view(-1, 128 * 14 * 14)
        x = nn.ReLU()(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Load the new trained model's state dict
new_cnn_model_path = cwd + "/models/type_cnn_model.pth"
new_cnn_model = NewCNNModel()
new_cnn_model.load_state_dict(torch.load(new_cnn_model_path, map_location=torch.device('cpu')), strict = False)
new_cnn_model.eval()

# Image preprocessing (same as the backend)
img_width, img_height = 224, 224
data_transforms = transforms.Compose([
    transforms.Resize((img_width, img_height)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Function to analyze image with both models and display results
def analyze_image(image, result_area):
    img = Image.open(image).convert('RGB')
    img_tensor = data_transforms(img).unsqueeze(0)  # Add batch dimension
    img_tensor = img_tensor.to(torch.device('cpu'))

    # Run the original model
    original_output = original_cnn_model(img_tensor)
    _, original_predicted_class = torch.max(original_output, 1)

    # Run the new model
    new_output = new_cnn_model(img_tensor)
    _, new_predicted_class = torch.max(new_output, 1)

    # Class names (adjust based on your dataset)
    class_names = ['dark', 'green', 'light', 'medium']

    # Get predictions from both models
    original_predicted_label = class_names[original_predicted_class.item()]
    new_predicted_label = class_names[new_predicted_class.item()]

    # Display the predictions
    result_area.write(f"Original model prediction: {original_predicted_label}")
    result_area.write(f"New model prediction: {new_predicted_label}")

# Streamlit app configuration
def config():
    st.set_page_config(
        layout="wide",
        page_title="Mr Beans | Beans",
        page_icon="☕️"
    )
    st.title("Beans Analyzer ☕️")
    st.info("Beans analyzer with Classification & Validity")

# Main view rendering
def render_view():
    left, right = st.columns(2)
    uploaded = left.file_uploader("Upload beans", type=['png', 'jpg', 'jpeg', 'webp'])
    enable = right.checkbox("Enable camera")
    picture = right.camera_input("Take the beans", disabled=not enable)

    new_left, new_right = st.columns(2)
    if uploaded:
        new_left.image(uploaded)
        analyze_image(uploaded, new_right)
    elif picture:
        new_left.image(picture)
        analyze_image(picture, new_right)

# Main entry point
if __name__ == "__main__":
    config()
    render_view()