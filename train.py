import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from model import ImageClassifier
from dataset import ImageAnnotationDataset

# Hyperparameters
batch_size = 32
learning_rate = 0.001
num_epochs = 10
img_height, img_width = 64, 64
class_names = ['chihuahua', 'japanese_spaniel', 'maltese']  # Replace with your class names

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset and DataLoader
data_transforms = transforms.Compose([
    transforms.Resize((img_height, img_width)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

dataset = ImageAnnotationDataset(
    image_folder = 'C:/Users/zawwi/Documents/ComputerVision/DogClassification/stanford_dogs/Images',
    annotation_folder = 'C:/Users/zawwi/Documents/ComputerVision/DogClassification/stanford_dogs/Annotation',
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

# Training Loop
for epoch in range(num_epochs):
    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Save the trained model
torch.save(model.state_dict(), "model.pth")
print("Model saved as model.pth")
