
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 1. Setup
torch.manual_seed(42)
device = torch.device("cpu")

print("Device:", device)

# 2. Load MNIST dataset
transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)

# 3. Build CNN model
class MNISTCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


model = MNISTCNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

print(model)

# 4. Training
epochs = 5

train_losses = []
train_accuracies = []

for epoch in range(epochs):
    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)

        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    epoch_loss = total_loss / total
    epoch_accuracy = 100 * correct / total

    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_accuracy)

    print(
        f"Epoch {epoch + 1}/{epochs} | "
        f"Train Loss: {epoch_loss:.4f} | "
        f"Train Accuracy: {epoch_accuracy:.2f}%"
    )

# 5. Plot training loss
plt.figure(figsize=(8, 5))
plt.plot(
    range(1, epochs + 1),
    train_losses,
    marker="o"
)
plt.title("CNN Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.xticks(range(1, epochs + 1))
plt.grid(True)
plt.tight_layout()
plt.savefig("cnn_loss_curve.png")
plt.show()

# 6. Plot training accuracy
plt.figure(figsize=(8, 5))
plt.plot(
    range(1, epochs + 1),
    train_accuracies,
    marker="o"
)
plt.title("CNN Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.xticks(range(1, epochs + 1))
plt.grid(True)
plt.tight_layout()
plt.savefig("cnn_accuracy_curve.png")
plt.show()

# 7. Evaluate on test data
model.eval()

correct = 0
total = 0
test_loss = 0

all_predictions = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        test_loss += loss.item() * labels.size(0)

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        all_predictions.extend(predictions.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

test_loss = test_loss / total
test_accuracy = 100 * correct / total

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.2f}%")

# 8. Plot confusion matrix
cm = confusion_matrix(all_labels, all_predictions)

cm_display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=range(10)
)

cm_display.plot(cmap="Blues", values_format="d")
plt.title("CNN Confusion Matrix")
plt.tight_layout()
plt.savefig("cnn_confusion_matrix.png")
plt.show()

# 9. Display sample predictions
images, labels = next(iter(test_loader))

model.eval()

with torch.no_grad():
    outputs = model(images.to(device))
    predictions = outputs.argmax(dim=1).cpu()

fig, axes = plt.subplots(2, 5, figsize=(12, 6))

for i, ax in enumerate(axes.flat):
    ax.imshow(images[i].squeeze(), cmap="gray")

    actual = labels[i].item()
    predicted = predictions[i].item()

    ax.set_title(
        f"Actual: {actual}\nPredicted: {predicted}"
    )
    ax.axis("off")

plt.tight_layout()
plt.savefig("cnn_sample_predictions.png")
plt.show()

# 10. Save trained model
torch.save(model.state_dict(), "mnist_cnn.pth")

print("Model saved successfully!")
print("Project completed!")
