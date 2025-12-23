import os
import csv
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from torchvision.models import ResNet18_Weights

# Paths 
DATA_DIR = "../data/processed/raw-img"
MODEL_PATH = "../models/resnet18_animals.pth"
OUT_DIR = "../reports"
OUT_CSV = os.path.join(OUT_DIR, "predictions.csv")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    print("Device:", DEVICE)
    os.makedirs(OUT_DIR, exist_ok=True)

    # Same transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Dataset
    full_ds = datasets.ImageFolder(DATA_DIR, transform=transform)
    class_names = full_ds.classes
    num_classes = len(class_names)
    print("Classes:", class_names)
    print("Total images:", len(full_ds))

    # Create a test split
    torch.manual_seed(42)
    test_size = int(0.2 * len(full_ds))
    train_size = len(full_ds) - test_size
    _, test_ds = random_split(full_ds, [train_size, test_size])

    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

    # Model (use new weights API to avoid warnings)
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    correct = 0
    total = 0

    # per-class counters
    per_class_total = [0] * num_classes
    per_class_correct = [0] * num_classes

    # store results for distributions
    rows = []
    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            logits = model(images)
            probs = softmax(logits)                    # probabilities
            conf, preds = torch.max(probs, dim=1)      # top-1 confidence & predicted class

            for i in range(labels.size(0)):
                y_true = labels[i].item()
                y_pred = preds[i].item()
                p = conf[i].item()
                is_correct = int(y_true == y_pred)

                rows.append([y_true, y_pred, p, is_correct])

                per_class_total[y_true] += 1
                per_class_correct[y_true] += is_correct

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    overall_acc = correct / total
    print(f"\nTest split size: {total}")
    print(f"Overall accuracy: {overall_acc:.4f}")

    print("\nPer-class accuracy:")
    for c in range(num_classes):
        if per_class_total[c] == 0:
            acc_c = 0.0
        else:
            acc_c = per_class_correct[c] / per_class_total[c]
        print(f"  {class_names[c]:<12}  {acc_c:.4f}   (n={per_class_total[c]})")

    # Save CSV for report + plotting
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["y_true", "y_pred", "top1_confidence", "correct"])
        writer.writerows(rows)

    print(f"\nSaved: {OUT_CSV}")
    print("Done.")

if __name__ == "__main__":
    main()
