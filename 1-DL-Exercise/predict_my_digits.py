import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os

# Define the CNN architecture
class ImprovedCNN(nn.Module):
    def __init__(self):
        super(ImprovedCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 1x28x28 → 32x28x28
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # 32x28x28 → 64x28x28
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)                          # 64x28x28 → 64x14x14

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # 64x14x14 → 128x14x14
        self.bn3 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)                           # 128x14x14 → 128x7x7

        self.dropout = nn.Dropout(0.4)

        self.fc1 = nn.Linear(128 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool2(x)

        x = self.dropout(x)
        x = x.view(-1, 128 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

# Load device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "improved_digit_model.pth")
model = ImprovedCNN().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
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
base_dir = os.path.dirname(os.path.abspath(__file__))

image_files = [
    os.path.join(base_dir, "handwritten_digits", "digit3.png"),
    os.path.join(base_dir, "handwritten_digits", "digit4.png"),
    os.path.join(base_dir, "handwritten_digits", "digit5.png")
]

# Run prediction for each image
for file in image_files:
    try:
        prediction = predict_image(file)
        print(f"Prediction for {file}: {prediction}")
    except Exception as e:
        print(f"Error processing {file}: {e}")