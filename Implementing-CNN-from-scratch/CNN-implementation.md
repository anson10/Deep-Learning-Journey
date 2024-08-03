# Table of Contents

1. Introduction
2. Convolution Operations
    - Cross-Correlation
    - Valid Cross-Correlation
3. Activation Functions
    - Sigmoid
4. Forward Pass in CNN
5. Backward Pass in CNN
6. Loss Functions
    - Binary Cross-Entropy Loss

# Key Concepts in CNNs: A Practical Guide

## 1. Introduction
Convolutional Neural Networks (CNNs) are a powerful class of deep neural networks primarily used for computer vision tasks. This guide focuses on the fundamental operations and concepts used in CNNs, illustrated through practical code examples.

## 2. Convolution Operations
Cross-correlation is a measure of similarity between a filter (kernel) and parts of the input image. In CNNs, it is used to extract features from the input image by sliding the filter over the image and computing the dot product.

### Valid Cross-Correlation
In valid cross-correlation, the filter slides only within the bounds of the input image, ensuring the output size is smaller than the input. Here’s a practical example:

```python
import numpy as np
from scipy import signal

class CNNLayer:
    def __init__(self, input_shape, kernel_size, depth):
        input_depth, input_height, input_width = input_shape
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1)
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)
        self.kernels = np.random.randn(*self.kernels_shape)
        self.biases = np.random.randn(*self.output_shape)

    def forward(self, input):
        self.input = input
        self.output = np.copy(self.biases)
        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output[i] += signal.correlate2d(self.input[j], self.kernels[i, j], "valid")
        return self.output
```
## 3. Activation Functions

### Sigmoid
The sigmoid function maps input values to a range between 0 and 1, making it suitable for binary classification tasks. It introduces non-linearity into the network, allowing it to learn complex patterns.

```python
class Sigmoid:
    def __init__(self):
        pass

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_prime(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)

    def forward(self, input):
        self.input = input
        self.output = self.sigmoid(self.input)
        return self.output

    def backward(self, output_gradient, learning_rate):
        return output_gradient * self.sigmoid_prime(self.input)

```

## 4. Forward Pass in CNN
The forward pass involves applying the convolution operation followed by the activation function. This helps in extracting features and introducing non-linearity.

```python
def forward(self, input):
    self.input = input
    self.output = np.copy(self.biases)
    for i in range(self.depth):
        for j in range(self.input_depth):
            self.output[i] += signal.correlate2d(self.input[j], self.kernels[i, j], "valid")
    return self.output
```

## 5. Backward Pass in CNN
The backward pass involves computing the gradients of the loss with respect to the filters and inputs. This is done using the chain rule and helps in updating the model parameters to minimize the loss.

```python
def backward(self, output_gradient, learning_rate):
    kernels_gradient = np.zeros(self.kernels_shape)
    input_gradient = np.zeros(self.input_shape)
    for i in range(self.depth):
        for j in range(self.input_depth):
            kernels_gradient[i, j] = signal.correlate2d(self.input[j], output_gradient[i], "valid")
            input_gradient[j] += signal.convolve2d(output_gradient[i], self.kernels[i, j], "full")

    self.kernels -= learning_rate * kernels_gradient
    self.biases -= learning_rate * output_gradient
    return input_gradient
```

## 6. Loss Functions

### Binary Cross-Entropy Loss
Binary Cross-Entropy Loss is used for binary classification tasks. It measures the performance of a classification model whose output is a probability value between 0 and 1.

```python
def binary_cross_entropy(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def binary_cross_entropy_prime(y_true, y_pred):
    return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true)

```





