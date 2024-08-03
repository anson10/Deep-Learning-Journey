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
        self.kernel_size = kernel_size
        self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1)
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)
        self.kernels = np.random.randn(*self.kernels_shape) * 0.1
        self.biases = np.random.randn(*self.output_shape) * 0.1  # Changed shape of biases

d   def forward(self, input):
        self.input = input
        self.output = np.zeros((input.shape[0], *self.output_shape))
        for b in range(input.shape[0]):  # Iterate over batch size
            for d in range(self.depth):
                for c in range(self.input_depth):
                    self.output[b, d] += signal.correlate2d(input[b, c], self.kernels[d, c], mode='valid')
                # Adding biases, now with correct shape
                self.output[b, d] += self.biases[d]
        return self.output
```
## 3. Activation Functions

### ReLU
ReLU (Rectified Linear Unit) is an activation function in neural networks that outputs the input directly if it's positive, and zero otherwise. It introduces non-linearity, allowing networks to learn complex patterns while being computationally efficient and helping to mitigate the vanishing gradient problem. Despite its simplicity, ReLU is highly effective and widely used in various neural network architectures.

```python
class ReLU:
    def forward(self, input):
        self.input = input
        return np.maximum(0, input)

    def backward(self, output_gradient, learning_rate):
        return output_gradient * (self.input > 0)

```

## 4. Forward Pass in CNN
The forward pass involves applying the convolution operation followed by the activation function. This helps in extracting features and introducing non-linearity.

```python
# Network setup
cnn_layer = CNNLayer(input_shape=(1, 28, 28), kernel_size=3, depth=16)
relu = ReLU()
flatten_layer = lambda x: x.reshape(x.shape[0], -1)
dense_layer = DenseLayer(16 * 26 * 26, num_classes)  # 16 filters, 26x26 output from CNN
softmax = Softmax()

# Training loop
for epoch in range(epochs):
    for i in range(0, len(x_train), batch_size):
        x_batch = x_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]
        
        # Forward pass
        conv_output = cnn_layer.forward(x_batch)
        relu_output = relu.forward(conv_output)
        flattened_output = flatten_layer(relu_output)
        dense_output = dense_layer.forward(flattened_output)
        softmax_output = softmax.forward(dense_output)
        loss = categorical_cross_entropy(y_batch, softmax_output)
        print(f"Epoch {epoch+1}, Batch {i//batch_size+1}, Loss: {loss}")

        # Backward pass
        loss_grad = categorical_cross_entropy_prime(y_batch, softmax_output)
        dense_grad = dense_layer.backward(loss_grad, learning_rate)
        flattened_grad = dense_grad.reshape(relu_output.shape)
        relu_grad = relu.backward(flattened_grad, learning_rate)
        conv_grad = cnn_layer.backward(relu_grad, learning_rate)

```

## 5. Backward Pass in CNN
The backward pass involves computing the gradients of the loss with respect to the filters and inputs. This is done using the chain rule and helps in updating the model parameters to minimize the loss.

## 6. Loss Functions

### Categotical Cross-Entropy Loss
Categorical Cross-Entropy is a loss function commonly used in multi-class classification problems where the target variable can take one of many possible categories. It measures the performance of a classification model whose output is a probability distribution over multiple classes.

```python
correct = 0
total = 0
for i in range(0, len(x_test), batch_size):
    x_batch = x_test[i:i+batch_size]
    y_batch = y_test[i:i+batch_size]
    
    conv_output = cnn_layer.forward(x_batch)
    relu_output = relu.forward(conv_output)
    flattened_output = flatten_layer(relu_output)
    dense_output = dense_layer.forward(flattened_output)
    softmax_output = softmax.forward(dense_output)
    
    predictions = np.argmax(softmax_output, axis=1)
    correct += np.sum(predictions == np.argmax(y_batch, axis=1))
    total += len(y_batch)


```



