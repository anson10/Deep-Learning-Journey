# Deep Learning Journey

Welcome to my Deep Learning Journey repository! This repository documents my exploration and learning in the field of deep learning. It includes various projects, experiments, and notes on key concepts and techniques in deep learning.

## Table of Contents

- [Introduction](#introduction)
- [Projects](#projects)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This repository is a collection of my deep learning projects and experiments. It includes implementations of different neural network architectures, solutions to common deep learning tasks, and notes on theoretical concepts. The goal is to build a comprehensive understanding of deep learning through hands-on practice and experimentation.

## Projects

### 1. MNIST Digit Classification
Implementation of a neural network to classify handwritten digits from the MNIST dataset using TensorFlow and Keras.
- **Folder**: `mnist-digit-classification`
- **Description**: Trains a neural network to classify images of handwritten digits.
- **Technologies**: TensorFlow, Keras, Python
- **Notebook**: [MNIST_Digit_Classification.ipynb](mnist-digit-classification/MNIST_Digit_Classification.ipynb)

### 2. MNIST Image Classification
Implementation of a neural network to classify fashion items from the MNIST dataset using TensorFlow and Keras.
- **Folder**: `mnist-image-classification`
- **Description**: Trains a neural network to classify images of fashion items.
- **Technologies**: TensorFlow, Keras, Python
- **Notebook**: [image-classifer.ipynb](mnist-image-classification/image-classifer.ipynb)

### 3. Loading Dataset using tensorflow API
Exploring `tf.data` API for creating efficient and scalable data input pipelines, including functions for reading, transforming, shuffling, batching, and prefetching datasets.
- **Folder**: `Loading-the-data-using-tf`
- **Description**: Going through a tf.data API.
- **Technologies**: TensorFlow, Keras, Python
- **Notebook**: [Loading_data_with_tf.ipynb](Loading-the-data-using-tf/Loading_data_with_tf.ipynb)


## Installation

To set up the environment and run the notebooks locally, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/anson10/deep-learning-journey.git
    cd deep-learning-journey
    ```

2. **Create a virtual environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install the required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the notebooks:

1. **Activate the virtual environment** (if not already activated):
    ```bash
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

2. **Launch Jupyter Notebook**:
    ```bash
    jupyter notebook
    ```

3. **Open the desired notebook** from the Jupyter interface and run the cells.

## Contributing

Contributions are welcome! If you have suggestions for improvements, new projects, or bug fixes, please open an issue or submit a pull request. Make sure to follow the existing code style and include relevant tests.

## License

This repository is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

Happy Learning!
