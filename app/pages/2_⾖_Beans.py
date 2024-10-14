import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import os

cwd = os.path.dirname(__file__) 

# roasted CNN Model used by the app
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
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

# class GradeCNNModel(nn.Module):
#     def __init__(self):
#         super(GradeCNNModel, self).__init__()
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
#         self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.fc1 = nn.Linear(128 * 14 * 14, 512)
#         self.dropout = nn.Dropout(0.5)
#         self.fc2 = nn.Linear(512, 3)  

#     def forward(self, x):
#         x = self.pool(nn.ReLU()(self.conv1(x)))
#         x = self.pool(nn.ReLU()(self.conv2(x)))
#         x = self.pool(nn.ReLU()(self.conv3(x)))
#         x = self.pool2(x)
#         x = x.view(-1, 128 * 14 * 14)
#         x = nn.ReLU()(self.fc1(x))
#         x = self.dropout(x)
#         x = self.fc2(x)
#         return x

# Load the roasted CNN model
roasted_cnn_model_path = cwd + "/models/roasted_cnn_model.pth"
roasted_cnn_model = CNNModel()
roasted_cnn_model.load_state_dict(torch.load(roasted_cnn_model_path, map_location=torch.device('cpu')), strict = False)
roasted_cnn_model.eval()

# Load the type trained model's state dict
type_cnn_model_path = cwd + "/models/type_cnn_model.pth"
type_cnn_model = CNNModel()
type_cnn_model.load_state_dict(torch.load(type_cnn_model_path, map_location=torch.device('cpu')), strict = False)
type_cnn_model.eval()

# Load the grade trained model's state dict
grade_cnn_model_path = cwd + "/models/type_cnn_model.pth"
# grade_cnn_model = GradeCNNModel()
grade_cnn_model = CNNModel()
grade_cnn_model.load_state_dict(torch.load(grade_cnn_model_path, map_location=torch.device('cpu')), strict = False)
grade_cnn_model.eval()


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

    # Run the roasted model
    roasted_output = roasted_cnn_model(img_tensor)
    _, roasted_predicted_class = torch.max(roasted_output, 1)

    # Run the type model
    type_output = type_cnn_model(img_tensor)
    _, type_predicted_class = torch.max(type_output, 1)

     # Run the grade model
    grade_output = grade_cnn_model(img_tensor)
    _, grade_predicted_class = torch.max(grade_output, 1)

    # Class names (adjust based on your dataset)
    roasted_class_names = ['Dark', 'Green', 'Light', 'Medium']
    type_class_names = ['Defect', 'Long Berry', 'Pea Berry', 'Premium']
    grade_class_names = ['Specialty','Premium','Exchange']

    # Get predictions from both models
    roasted_predicted_label = roasted_class_names[roasted_predicted_class.item()]
    type_predicted_label = type_class_names[type_predicted_class.item()]
    grade_predicted_label = grade_class_names[grade_predicted_class.item()]

    # Display the predictions
    result_area.markdown(f"**Roasted:**")
    result_area.write({roasted_predicted_label})
    result_area.markdown(f"**Type:**")
    result_area.write({type_predicted_label})
    result_area.markdown(f"**Grade:**")
    result_area.write({grade_predicted_label})
        
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

    type_left, type_right = st.columns(2)
    if uploaded:
        type_left.image(uploaded)
        analyze_image(uploaded, type_right)
    elif picture:
        type_left.image(picture)
        analyze_image(picture, type_right)

# Main entry point
if __name__ == "__main__":
    config()
    render_view()