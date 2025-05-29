import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import matplotlib.pyplot as plt

# ✅ Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🧠 Using device:", device)

# ✅ Data transforms with augmentation
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ✅ Load dataset
dataset = datasets.ImageFolder("dataset", transform=transform)
class_names = dataset.classes
print(f"✅ Classes: {class_names}")

# ✅ Split dataset
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# ✅ Model setup
model = models.resnet34(weights="DEFAULT")
model.fc = nn.Linear(model.fc.in_features, 2)
model.to(device)

# ✅ Optimizer & scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

# ✅ Tracking metrics
train_losses = []
val_accuracies = []

num_epochs = 15
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    loop = tqdm(train_loader, desc=f"🔁 Epoch {epoch+1}/{num_epochs}", leave=False)

    for inputs, labels in loop:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    train_losses.append(running_loss)

    # ✅ Evaluate
    model.eval()
    correct = 0
    total = 0
    class_correct = [0] * 2
    class_total = [0] * 2

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Per-class accuracy
            for i in range(len(labels)):
                label = labels[i]
                class_total[label] += 1
                class_correct[label] += (predicted[i] == label).item()

    acc = 100 * correct / total
    val_accuracies.append(acc)
    print(f"\n📘 Epoch {epoch+1}/{num_epochs}, Loss: {running_loss:.2f}, Validation Accuracy: {acc:.2f}%")

    for i in range(2):
        if class_total[i] > 0:
            print(f"   🔹 {class_names[i]} Accuracy: {100 * class_correct[i] / class_total[i]:.2f}%")

    scheduler.step()

# ✅ Save model
os.makedirs("model", exist_ok=True)
torch.save(model.state_dict(), "model/fakevision_model.pt")
print("\n✅ Training complete and model saved!")

# ✅ Plot loss and accuracy
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.title("Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(val_accuracies, label='Validation Accuracy')
plt.title("Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.grid()
plt.legend()

plt.tight_layout()
plt.savefig("model/training_metrics.png")
plt.show()