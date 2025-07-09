# backend/model_handler.py

import torch
import torchvision.transforms as transforms
from torchvision.transforms.functional import invert
from torchvision import models
import torch.nn as nn
from PIL import Image
import io
import json
import os 
import requests

#1.Define Model Loading Instructions 

print("Defining model loading instructions...")

# Define the MNIST CNN class here so the loader functions can access it
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

def load_mnist_model():
    model = MNIST_CNN()
    
    dir_path = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(dir_path, '..', 'models', 'mnist_cnn.pth')
    
    # Load the model using the new, absolute path
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    return model

model_loaders = {
    "resnet18": lambda: models.resnet18(weights=models.ResNet18_Weights.DEFAULT),
    "mobilenet_v2": lambda: models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT),
    "mnist_cnn": load_mnist_model
}

# This dictionary will act as a cache for models once they are loaded
loaded_models = {}
print("Model loaders are ready.")


#2. Image Transformations 
imagenet_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

mnist_transform = transforms.Compose([
    transforms.Lambda(invert),
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

#3. Load Labels 
imagenet_class_index = None
try:
    dir_path = os.path.dirname(os.path.realpath(__file__))
    
    file_path = os.path.join(dir_path, 'data', 'imagenet_class_index.json')
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    imagenet_class_index = {int(k): v[1] for k, v in data.items()}
    print("ImageNet labels loaded from local file.")

except Exception as e:
    print(f"Could not load ImageNet labels from {file_path}: {e}")

mnist_labels = [str(i) for i in range(10)]


#4. Prediction Function 
def get_prediction(model_name, image_bytes):
    try:
        if model_name not in loaded_models:
            print(f"Loading model '{model_name}' for the first time...")
            model = model_loaders[model_name]()
            model.eval()
            loaded_models[model_name] = model
            print(f"Model '{model_name}' loaded and cached.")
        
        # Get the model from the cache
        model = loaded_models[model_name]
        
        # The rest of the prediction logic is the same
        image = Image.open(io.BytesIO(image_bytes))

        if model_name in ["resnet18", "mobilenet_v2"]:
            if image.mode != "RGB":
                image = image.convert("RGB")
            tensor = imagenet_transform(image).unsqueeze(0)
            labels = imagenet_class_index
        elif model_name == "mnist_cnn":
            if image.mode != "RGB":
                image = image.convert("RGB")
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