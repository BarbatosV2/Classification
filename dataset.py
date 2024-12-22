import os
from torch.utils.data import Dataset
from PIL import Image
from utils import parse_annotation, class_to_index
from pathlib import Path

class ImageAnnotationDataset(Dataset):
    def __init__(self, image_folder, annotation_folder, class_names, transform=None):
        self.image_paths = []
        self.annotation_paths = []
        self.class_names = class_names
        self.transform = transform

        # Build a mapping from image paths to annotation paths
        for root, _, files in os.walk(image_folder):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):  # Handle case-insensitive extensions
                    image_path = Path(root) / file
                    class_folder = Path(root).name  # Get class folder name
                    annotation_file = file.rsplit('.', 1)[0]  # Remove file extension, no need to add '.file'

                    # Build the full annotation file path
                    annotation_path = Path(annotation_folder) / class_folder / annotation_file

                    print(f"Checking image: {image_path}")
                    print(f"Looking for annotation: {annotation_path}")

                    if annotation_path.exists():  # Only add if annotation exists
                        self.image_paths.append(str(image_path))
                        self.annotation_paths.append(str(annotation_path))
                    else:
                        print(f"Annotation file not found for image: {image_path}")

        if len(self.image_paths) == 0:
            print("No valid samples found in the dataset!")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        # Load annotation and get label
        annotation_path = self.annotation_paths[idx]
        label = parse_annotation(annotation_path)

        # Convert label to numerical index
        label_idx = class_to_index(label, self.class_names)

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        return image, label_idx
