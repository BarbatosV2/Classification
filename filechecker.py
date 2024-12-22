import os

# Path to the specific folder
annotation_folder_path = r"C:\Users\zawwi\Documents\ComputerVision\DogClassification\stanford_dogs\Annotation\n02085620-Chihuahua"

# Check if the specific file exists
file_name = "n02085620_199"  # File to check
file_path = os.path.join(annotation_folder_path, file_name)

if os.path.exists(file_path):
    print(f"File found: {file_path}")
else:
    print(f"File not found: {file_path}")
