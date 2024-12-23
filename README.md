# Image Classification

# Create a directory for the dataset
```
mkdir -p stanford_dogs
cd stanford_dogs
```

# Download the dataset
Depends on what dataset u want to train. For me, I am training dog dataset.
```
wget http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar
wget http://vision.stanford.edu/aditya86/ImageNetDogs/annotation.tar
```

If not working, can directly download at http://vision.stanford.edu/aditya86/ImageNetDogs/ 

# Extract the images
```
tar -xvf images.tar
tar -xvf annotation.tar
```
# Read Me

Edit or add or modify **class_name** from **train.py** and **check.py** depends on how many classes in the training set.

```batch_size```, ```img_height```, ```img_weight``` and ```learning_rate``` can be changed to speed up or preserve vram or gpu.

If ```img_height```, ```img_weight``` is changed, need to modify the models as well. 

File structures might need to change for Linux. 

# How to Run

# To Train

 ```
py train.py
```

# Run trained Data

This will used the trained pth called model_final.pth 
```
py check.py imagefile_location/image.jpg
```

# Final Result of Training (Epoch 20)

![training_graph](https://github.com/user-attachments/assets/f121359c-6426-4d34-b8c0-1c11a1ee69c6)
