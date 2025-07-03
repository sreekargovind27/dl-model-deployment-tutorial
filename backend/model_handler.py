# backend/model_handler.py

import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
from PIL import Image
import io
import json

# --- 1. Load Pre-trained Models ---
# Load models and set to evaluation mode

# We use a dictionary to easily access them by name
print("Loading pre-trained models...")
pretrained_models = {
    "resnet18": models.resnet18(weights=models.ResNet18_Weights.DEFAULT),
    "mobilenet_v2": models.mobilenet_v2(weights=models.MobileNetV2_Weights.DEFAULT)
}
for model in pretrained_models.values():
    model.eval()
print("Pre-trained models loaded.")

