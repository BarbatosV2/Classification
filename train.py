import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from model import ImageClassifier
from dataset import ImageAnnotationDataset
import matplotlib.pyplot as plt
import os

# Hyperparameters
batch_size = 32
learning_rate = 0.001
num_epochs = 10
img_height, img_width = 64, 64
class_names = ['Chihuahua', 'Japanese_spaniel', 'Maltese_dog']  # Replace with your class names

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset and DataLoader
data_transforms = transforms.Compose([
    transforms.Resize((img_height, img_width)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

dataset = ImageAnnotationDataset(
    image_folder='C:/Users/zawwi/Documents/ComputerVision/DogClassification/stanford_dogs/Images',
    annotation_folder='C:/Users/zawwi/Documents/ComputerVision/DogClassification/stanford_dogs/Annotation',
    class_names=class_names,
    transform=data_transforms
)

print(f"Number of samples in dataset: {len(dataset)}")
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model, Loss, Optimizer
num_classes = len(class_names)
model = ImageClassifier(num_classes=num_classes).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Prepare to save graphs and models
if not os.path.exists("train_graph"):
    os.makedirs("train_graph")
if not os.path.exists("model"):
    os.makedirs("model")

# Initialize lists to track loss and accuracy
epoch_losses = []
epoch_accuracies = []

# Training Loop
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    
    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update running loss
        running_loss += loss.item()

        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_predictions += labels.size(0)

    # Calculate average loss and accuracy for the epoch
    avg_loss = running_loss / len(data_loader)
    accuracy = correct_predictions / total_predictions * 100

    # Save loss and accuracy for plotting
    epoch_losses.append(avg_loss)
    epoch_accuracies.append(accuracy)

    # Print epoch stats
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    # Save model after each epoch
    torch.save(model.state_dict(), f"model/model_epoch_{epoch+1}.pth")
    print(f"Model saved as model_epoch_{epoch+1}.pth")

# Plot Loss and Accuracy
plt.figure(figsize=(12, 6))

# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), epoch_losses, label="Loss", color="blue")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss per Epoch")
plt.grid(True)

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), epoch_accuracies, label="Accuracy", color="green")
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy per Epoch")
plt.grid(True)

# Save the graphs
plt.tight_layout()
plt.savefig("train_graph/training_graph.png")
plt.show()

# Final model save
torch.save(model.state_dict(), "model/model_final.pth")
print("Final model saved as model_final.pth")
