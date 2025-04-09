import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os

# Define the CNN architecture (must match the one used in training)
class ImprovedCNN(nn.Module):
    def __init__(self):
        super(ImprovedCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))      
        x = F.relu(self.conv2(x))      
        x = self.pool(x)               
        x = self.dropout(x)
        x = x.view(-1, 64 * 14 * 14)   
        x = F.relu(self.fc1(x))        
        x = self.dropout(x)
        x = F.relu(self.fc2(x))        
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

# Load device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
model = ImprovedCNN().to(device)
model.load_state_dict(torch.load("improved_digit_model.pth", map_location=device))
model.eval()

# Image transformation (must match training transforms)
transform = transforms.Compose([
    transforms.Grayscale(),                    # Convert to grayscale
    transforms.Resize((28, 28)),               # Resize to 28x28
    transforms.ToTensor(),                     # Convert to tensor
    transforms.Normalize((0.5,), (0.5,))       # Normalize
])

# Prediction function
def predict_image(img_path):
    image = Image.open(img_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)  # Shape: [1, 1, 28, 28]
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return predicted.item()

# List of image filenames in handwritten_digits folder
image_files = ["handwritten_digits/digit3.png", 
               "handwritten_digits/digit4.png", 
               "handwritten_digits/digit5.png"]

# Run prediction for each image
for file in image_files:
    try:
        prediction = predict_image(file)
        print(f"Prediction for {file}: {prediction}")
    except Exception as e:
        print(f"Error processing {file}: {e}")