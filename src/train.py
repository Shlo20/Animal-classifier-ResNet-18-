import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

def main():
    # Paths
    DATA_DIR = "../data/processed/raw-img"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("Device:", DEVICE)

    if torch.cuda.is_available():
        print("GPU name:", torch.cuda.get_device_name(0))
        print("CUDA version (torch):", torch.version.cuda)

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Dataset
    dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
    num_classes = len(dataset.classes)

    print("Classes:", dataset.classes)
    print("Number of classes:", num_classes)
    print("Total images:", len(dataset))

    train_loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0  # for Windows
    )

    # Model
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(DEVICE)

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    EPOCHS = 5

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {running_loss/len(train_loader):.4f}")

    # Save model
    os.makedirs("../models", exist_ok=True)
    torch.save(model.state_dict(), "../models/resnet18_animals.pth")

    print(" Training complete. Model saved.")

if __name__ == "__main__":
    main()
