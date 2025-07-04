# backend/model_handler.py

import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
from PIL import Image
import io
import json

# Load Pre-trained Models 
# Load models and set to evaluation mode

# We use a dictionary to easily access them by name
print("Loading pre-trained models...")
pretrained_models = {
    "resnet18": models.resnet18(weights=models.ResNet18_Weights.DEFAULT),
    "mobilenet_v2": models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
}
for model in pretrained_models.values():
    model.eval()
print("Pre-trained models loaded.")

# 2. Define and Load Custom MNIST Model 

# Define the same simple CNN structure used for training
class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc_drop = nn.Dropout()
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = torch.relu(torch.max_pool2d(self.conv1(x), 2))
        x = torch.relu(torch.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = torch.relu(self.fc1(x))
        x = self.fc_drop(x)
        x = self.fc2(x)
        return torch.log_softmax(x, dim=1)

# Model Loading
print("Loading MNIST model with trained weights...")

# Path to the trained model weights file created by train_mnist.py
MODEL_PATH = "../models/mnist_cnn.pth"

# Initialize the model structure
mnist_model = MNIST_CNN()

mnist_model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))

#setting model to evaluation mode
mnist_model.eval() 
print("MNIST model loaded successfully.")

#addinig mnist model to the dictionary
pretrained_models["mnist_cnn"] = mnist_model

# 3. Image Transformations 

# ImageNet models expect 3-channel (RGB) 224x224 images
imagenet_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# MNIST model expects 1-channel (grayscale) 28x28 images
mnist_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

''' 
Load Labels 
ImageNet labels
We'll download this file in our app startup
'''

imagenet_class_index = None
try:
    import requests
    response = requests.get("https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json")
    data = response.json()
    imagenet_class_index = {int(k): v[1] for k, v in data.items()}
    print("ImageNet labels loaded.")
except Exception as e:
    print(f"Could not load ImageNet labels: {e}")


# MNIST labels are just the digits 0-9
mnist_labels = [str(i) for i in range(10)]

# 5. Prediction Function 
def get_prediction(model_name, image_bytes):

    try:
        model = pretrained_models[model_name]
        image = Image.open(io.BytesIO(image_bytes))

        if model_name in ["resnet18", "mobilenet_v2"]:
            # Ensure image is RGB for ImageNet models
            if image.mode != "RGB":
                image = image.convert("RGB")
            tensor = imagenet_transform(image).unsqueeze(0)
            labels = imagenet_class_index
        elif model_name == "mnist_cnn":
            tensor = mnist_transform(image).unsqueeze(0)
            labels = mnist_labels
        else:
            return "Unknown model"

        with torch.no_grad():
            output = model(tensor)
            _, predicted_idx = torch.max(output, 1)
            prediction = labels[predicted_idx.item()]
            return prediction.replace("_", " ").title()

    except Exception as e:
        print(f"Error during prediction: {e}")
        return "Error making prediction"