# MNIST Handwritten Digit Classification Using CNN

## Project Overview
This project implements a Convolutional Neural Network (CNN) using PyTorch to classify handwritten digits from the MNIST dataset.

The model learns visual features such as edges, curves, and patterns through convolutional layers, then predicts the digit represented by each image.

## Objectives
- Understand the fundamentals of Convolutional Neural Networks.
- Apply convolution, ReLU activation, and max pooling.
- Train and evaluate a CNN using PyTorch.
- Visualize training performance and classification results.
- Save the trained model for future use.

## Dataset
The project uses the MNIST handwritten digit dataset.

- Training images: 60,000
- Test images: 10,000
- Image size: 28 × 28 pixels
- Image channels: 1 (grayscale)
- Number of classes: 10 (digits 0–9)

## Model Architecture
The CNN consists of the following layers:

1. Convolutional layer: 1 input channel and 16 output channels.
2. ReLU activation.
3. Max pooling.
4. Convolutional layer: 16 input channels and 32 output channels.
5. ReLU activation.
6. Max pooling.
7. Flatten layer.
8. Fully connected layer with 128 neurons.
9. ReLU activation.
10. Output layer with 10 classes.

## Technologies Used
- Python
- PyTorch
- Torchvision
- Matplotlib
- Scikit-learn

## Training Configuration
- Epochs: 5
- Batch size: 64
- Optimizer: Adam
- Learning rate: 0.001
- Loss function: Cross
