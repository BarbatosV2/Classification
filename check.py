import argparse
import torch
from torchvision import transforms
from PIL import Image
from model import ImageClassifier

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
class_names = ['Chihuahua', 'Japanese_spaniel', 'Maltese_dog']  # class names
num_classes = len(class_names)
model = ImageClassifier(num_classes=num_classes).to(device)
model.load_state_dict(torch.load("model/model_final.pth", map_location=device))
model.eval()

# Define transformation
data_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def classify_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = data_transforms(image).unsqueeze(0).to(device)

    # Predict
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    class_name = class_names[predicted.item()]

    return class_name

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Classify an image using a trained model.")
    parser.add_argument("image_path", type=str, help="Path to the image to be classified.")
    args = parser.parse_args()

    # Classify the image
    class_name = classify_image(args.image_path)
    print(f"Predicted class: {class_name}")
