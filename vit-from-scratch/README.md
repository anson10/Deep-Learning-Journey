# Vision Transformers (ViTs) with MNIST Dataset

## Overview

This project implements a Vision Transformer (ViT) model for image classification using the MNIST dataset. Vision Transformers are a novel deep learning architecture that has shown promising results in computer vision tasks, sometimes outperforming traditional Convolutional Neural Networks (CNNs).

## Vision Transformers (ViTs)

### What are Vision Transformers?

Vision Transformers (ViTs) are an adaptation of the Transformer architecture, originally designed for natural language processing tasks, to the domain of image processing. Unlike CNNs, which rely on convolutional layers to capture local patterns in images, ViTs process images by splitting them into patches and treating each patch as a sequence token. These tokens are then processed by the Transformer, which captures the relationships between different patches.

### Key Components of ViTs:

- **Patch Embedding:** The input image is divided into fixed-size patches, which are flattened and linearly embedded into vectors.
- **Positional Encoding:** Since Transformers do not have any built-in notion of the order of tokens, positional encodings are added to the patch embeddings to retain spatial information.
- **Transformer Encoder:** The core of the ViT, consisting of multiple layers of multi-head self-attention and feed-forward neural networks.
- **Classification Head:** The final layer, which classifies the image based on the output of the Transformer encoder.

### Why Use ViTs?

ViTs have the advantage of being more flexible and scalable compared to CNNs, as they do not require hand-crafted convolutional filters. They can learn global image relationships more effectively, especially in large datasets, making them suitable for a wide range of computer vision tasks.

## Dataset

The project uses the **MNIST dataset**, a widely-used dataset in the field of image processing. MNIST consists of 70,000 grayscale images of handwritten digits (0-9), with 60,000 images for training and 10,000 images for testing.

- **Training Data:** 60,000 images.
- **Validation Data:** Split from the training set (typically 10,000 images).
- **Test Data:** 10,000 images.

The images are 28x28 pixels, with pixel values ranging from 0 to 255.

## Project Structure

- **MNISTTrainDataset:** A custom PyTorch `Dataset` class for training data, with image transformations such as random rotations and normalization.
- **MNISTValDataset:** A custom PyTorch `Dataset` class for validation data, with basic normalization.
- **MNISTTestDataset:** A custom PyTorch `Dataset` class for test data, used to evaluate the model.

### Code Explanation

- **Transformations:** 
  - **`transforms.ToPILImage()`**: Converts a tensor to a PIL Image.
  - **`transforms.RandomRotation(15)`**: Randomly rotates the image by 15 degrees.
  - **`transforms.ToTensor()`**: Converts a PIL Image or numpy array to a tensor.
  - **`transforms.Normalize([0.5], [0.5])`**: Normalizes the image with mean 0.5 and standard deviation 0.5.

- **Model Training and Validation:** 
  - The model is trained using the Cross-Entropy Loss and optimized using Adam with a learning rate, betas, and weight decay specified.
  - The training and validation losses and accuracies are calculated and printed for each epoch.

### DataLoaders

- **`train_dataloader`**: Loads training data in batches, shuffling the data at every epoch.
- **`val_dataloader`**: Loads validation data in batches.
- **`test_dataloader`**: Loads test data in batches for model evaluation.

### Training Loop

The training loop runs for a specified number of epochs, calculating the loss and accuracy on both the training and validation datasets.

```python
for epoch in range(EPOCHS):
    # Training
    model.train()
    for img_label in train_dataloader:
        img = img_label["image"].float().to(device)
        label = img_label["label"].type(torch.uint8).to(device)
        ...
    # Validation
    model.eval()
    with torch.no_grad():
        for img_label in val_dataloader:
            img = img_label["image"].float().to(device)
            label = img_label["label"].type(torch.uint8).to(device)
            ...
```

## Requirements

To run this project, you will need to install the following dependencies:

- `torch`
- `torchvision`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `numpy`
- `tqdm`

You can install the dependencies using the following command:

```bash
pip install -r requirements.txt
```

## How to run

1. Clone the repository.

2. Install the dependencies listed in requirements.txt.

3. Run the ViT.ipynb notebook
