import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
import io
import os 

cwd = os.path.dirname(__file__) 

class CNNModel(nn.Module):
    def _init_(self):
        super(CNNModel, self)._init_()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Additional pooling to further reduce dimensionality
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 14 * 14, 512)  
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 4)  # Adjust based on number of classes

    def forward(self, x):
        x = self.pool(nn.ReLU()(self.conv1(x)))
        x = self.pool(nn.ReLU()(self.conv2(x)))
        x = self.pool(nn.ReLU()(self.conv3(x)))
        x = self.pool2(x)  # Additional pooling layer for reducing dimensions
        x = x.view(-1, 128 * 14 * 14)
        x = nn.ReLU()(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Initialize the Flask app and enable CORS
app = Flask(_name_)
CORS(app)



# Load the CNN model
cnn_model_path = cwd + "/models/roasted_cnn_model.pth"
cnn_model = CNNModel()

def load_model():
    global cnn_model
    # Load model state
    cnn_model.load_state_dict(torch.load(cnn_model_path, map_location=torch.device('cpu')))
    cnn_model.eval()  # Set model to evaluation mode

# Image preprocessing: ensure consistency with your training process
img_width, img_height = 224, 224
data_transforms = transforms.Compose([
    transforms.Resize((img_width, img_height)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Endpoint to analyze an image
@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Read image file and preprocess it
    try:
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        img_tensor = data_transforms(img).unsqueeze(0)  # Add batch dimension

        # Ensure tensor is on the correct device (CPU)
        img_tensor = img_tensor.to(torch.device('cpu'))

        # Run the model on the input image
        output = cnn_model(img_tensor)
        _, predicted_class = torch.max(output, 1)
        
        # Assuming class names based on your dataset
        class_names = ['Class_1', 'Class_2', 'Class_3', 'Class_4']  # Adjust class names
        predicted_label = class_names[predicted_class.item()]

        # Return the predicted class as JSON
        return jsonify({'predicted_class': predicted_label})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if _name_ == '_main_':
    load_model()  # Load the trained model before running the app
    app.run(debug=True)