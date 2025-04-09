import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Using CPU")

# Define CNN model with one convolutional and one pooling layer
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # intput 28x28 (digit data)
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5) # one channel and 10 kernels
        # output size from Conv2d: 24 x 24 (Width= [(28-5+0)/1 + 1])
        self.pool = nn.MaxPool2d(2, 2)
        # output size from MaxPool2d: 25/2 = 12
        # if flattened input size = 10 x 12 x 12 = 1440
        self.fc1 = nn.Linear(1440, 50) # 50 neurons in the first layer
        self.fc2 = nn.Linear(50, 10) # output 10 neurons
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 1440)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Data transformation
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Load dataset
train_dataset = datasets.MNIST(root='', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='', train=False, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

# Initialize model, loss, and optimizer
model = CNN()
model = CNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Training loop
for epoch in range(10):
    model.train()
    total_loss = 0
    for images, labels in train_loader: # load one batch at a time
        images, labels = images.to(device), labels.to(device) # copying data to GPU
        images = images.view(-1, 1, 28, 28) # reshape the image tensor
        optimizer.zero_grad() # resetting gradients
        output = model(images) # forward pass, automatically calling forward(x)
        loss = criterion(output, labels)
        loss.backward() # backpropagation
        optimizer.step() # weight updates
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

# Evaluate accuracy
correct, total = 0, 0
model.eval()
with torch.no_grad(): # disabling gradient tracking
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device) # copying data to GPU
        
        images = images.view(-1, 1, 28, 28)
        output = model(images) # predict output class of image input
        # max(output,1) extract the class prediction from the output
        # by finding the class that has the highest score
        _, predicted = torch.max(output, 1) # 1 specifies the class dimension (column)
        total += labels.size(0) # number of samples in the current batch
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {correct / total * 100:.2f}%")

# Save the model parameters that can be loaded later
torch.save(model.state_dict(), "my_cnn.ph")
