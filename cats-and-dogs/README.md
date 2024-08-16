# Cats and Dogs Classification using CNN

This project uses a Convolutional Neural Network (CNN) to classify images of cats and dogs. The goal is to develop a model that can accurately distinguish between the two types of animals based on the provided image data.

## Project Overview

### 1. Dataset
The dataset is organized into two main directories: `train` and `test1`. Each of these directories contains images of both cats and dogs.

- Dataset Link : https://www.kaggle.com/c/dogs-vs-cats

- **`train/`**: Contains the training data.
  - All images are currently mixed together in this folder.
- **`test1/`**: Contains the testing data.
  - All images are currently mixed together in this folder.

### 2. Data Organization
To effectively train the CNN model, the images need to be organized into separate subdirectories for cats and dogs within both the `train` and `test1` directories.

#### Example Structure After Organization:
```
dogs-and-cats/
│
├── train/
│ ├── cats/
│ │ ├── cat.0.jpg
│ │ ├── cat.1.jpg
│ │ └── ...
│ ├── dogs/
│ │ ├── dog.0.jpg
│ │ ├── dog.1.jpg
│ │ └── ...
│
├── test1/
│ ├── cats/
│ │ ├── cat.2.jpg
│ │ ├── cat.3.jpg
│ │ └── ...
│ ├── dogs/
│ │ ├── dog.2.jpg
│ │ ├── dog.3.jpg
│ │ └── 
```

### 3. Model Training
Once the data is organized, the CNN model can be trained to classify the images. The model will learn to distinguish between cats and dogs by analyzing the patterns in the images.

## Conclusion
This project demonstrates a simple yet effective approach to image classification using CNNs. Proper data organization is crucial for the success of the model. Once the images are sorted, the model can be trained to achieve high accuracy in distinguishing between cats and dogs.
