# Swayam Shree

import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def main():
    ################################################################
    # 1. DATA LOADING & PREPROCESSING
    ################################################################
    
    data_dir = "3-Image-Classification/L05_DL_Vision_Dogs"  # Folder containing train/val subfolders:
                                    # L05_DL_Vision_Dogs/
                                    #    train/
                                    #       Bulldog/
                                    #       Poodle/
                                    #       ...
                                    #    val/
                                    #       Bulldog/
                                    #       Poodle/
                                    #       ...
    
    # Standard ImageNet-like transforms:
    # - Resize, crop, flip, normalize with ImageNet means/std
    train_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    
    train_dir = os.path.join(data_dir, "train")
    val_dir   = os.path.join(data_dir, "val")
    
    # Create datasets from ImageFolder
    train_dataset = torchvision.datasets.ImageFolder(
        root=train_dir,
        transform=train_transforms
    )
    
    val_dataset = torchvision.datasets.ImageFolder(
        root=val_dir,
        transform=val_transforms
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)
    
    num_classes = len(train_dataset.classes)  # # of dog breeds (e.g. 3 if Bulldog, Poodle, Beagle)
    
    ################################################################
    # 2. LOAD PRETRAINED MODEL (RESNET) AND MODIFY
    ################################################################
    model = torchvision.models.resnet18(pretrained=True)
    
    # Print the model architecture
    print("\n================== ResNet Model Architecture ==================")
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\nTotal parameters in model:       ", total_params)
    print("Trainable parameters in model:  ", trainable_params)

    # Replace the final fully-connected layer to match our dog-breed count
    in_features = model.fc.in_features  # Typically 512 for ResNet18
    model.fc = nn.Linear(in_features, num_classes)
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    ################################################################
    # 3. DEFINE LOSS & OPTIMIZER
    ################################################################
    criterion = nn.CrossEntropyLoss()
    # Here we only fine-tune the final layer. If you want to fine-tune more,
    # unfreeze them similarly and add them to optimizer.
    optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)
    
    ################################################################
    # 4. TRAINING LOOP
    ################################################################
    num_epochs = 20
    print(f"\n===== Starting Training for {num_epochs} epoch(s) =====")
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
    
    end_time = time.time()
    total_training_time = end_time - start_time
    print(f"===== Training complete. Time elapsed: {total_training_time:.2f} seconds =====\n")
    
    ################################################################
    # 5. EVALUATION / VALIDATION
    ################################################################
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            # Get predicted class with highest logit
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100.0 * correct / total
    print(f"Validation Accuracy: {accuracy:.2f}%")
    
    # If you only want to measure pure inference time, measure it separately.
    total_time = end_time - start_time
    print(f"Total classification time (train + val): {total_time:.2f} seconds")

    ################################################################
    # 6. PRINT ANY ADDITIONAL ARCHITECTURAL COMPONENTS
    ################################################################
    # Example: print the named children, which can help identify extra blocks
    print("\nAdditional model components not discussed in detail might include:")
    for name, module in model.named_children():
        print(f" - {name}: {module.__class__.__name__}")

if __name__ == "__main__":
    main()
