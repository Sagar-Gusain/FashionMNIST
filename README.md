# FashionMNIST Image Classification with PyTorch, Optuna & VGG16

A deep learning project that classifies FashionMNIST images using Convolutional Neural Networks (CNNs) and Transfer Learning with PyTorch. This project explores the impact of hyperparameter optimization using **Optuna** and pretrained models using **VGG16** to improve image classification performance.

## Project Overview

The project explores three approaches to FashionMNIST classification:

1. **Baseline CNN:** A CNN trained using manually selected hyperparameters.
2. **Optuna-Optimized CNN:** A CNN whose hyperparameters are optimized using Optuna to improve validation performance.
3. **VGG16 Transfer Learning:** A pretrained VGG16 model adapted for FashionMNIST classification using transfer learning and feature extraction.

The project demonstrates the complete deep learning workflow, including data preprocessing, model development, training, evaluation, hyperparameter optimization, and transfer learning.

## Features

- CNN implemented using PyTorch
- FashionMNIST dataset from Torchvision
- Image preprocessing using Transforms
- DataLoader for batch processing
- Model training and evaluation
- Hyperparameter optimization using Optuna
- Transfer Learning using pretrained VGG16
- Feature extraction using frozen convolutional layers
- Custom classifier for FashionMNIST's 10 classes
- Visualization of optimization results

## Tech Stack

- Python
- PyTorch
- Torchvision
- Optuna
- NumPy
- Matplotlib
- Google Colab

## Dataset

The project uses the **FashionMNIST** dataset containing 70,000 grayscale images (28×28 pixels) across 10 clothing categories.

- 60,000 training images
- 10,000 test images
- 10 classes

For VGG16, images are converted into three-channel images and resized to match the pretrained model's input requirements.

## Model 1: Baseline CNN

The first model uses manually selected hyperparameters and a CNN architecture consisting of convolutional, pooling, and fully connected layers.

### Results

| Metric | Result |
|---|---:|
| Training Loss | 0.4518 |
| Test Accuracy | **83.92%** |

## Model 2: Optuna Hyperparameter Optimization

Optuna was used to search for better hyperparameters and improve CNN performance.

### Hyperparameters Tuned

| Hyperparameter | Search Space |
|---|---|
| Number of Filters | 16, 32, 64 |
| Hidden Neurons | 128, 256, 512, 1024 |
| Dropout | 0.1 – 0.5 |
| Learning Rate | 0.0001 – 0.01 |
| Batch Size | 32, 64, 128 |

### Optimization Setup

- Optimization Framework: Optuna
- Sampling Strategy: TPE (Tree-structured Parzen Estimator)
- Number of Trials: 20
- Objective: Maximize validation accuracy
- Training Epochs per Trial: 5

### Results

| Metric | Result |
|---|---:|
| Best Validation Accuracy | **92.15%** |
| Number of Trials | 20 |

The best hyperparameter combination was selected using Optuna's optimization process.

## Model 3: VGG16 Transfer Learning

Transfer learning was implemented using **VGG16 pretrained on ImageNet**.

Instead of training a CNN from scratch, the pretrained convolutional layers were reused as a feature extractor.

### Implementation

- Loaded pretrained VGG16 weights using Torchvision.
- Converted FashionMNIST grayscale images into three-channel images.
- Resized images for compatibility with VGG16.
- Froze the pretrained convolutional layers to preserve learned features.
- Replaced the original 1000-class classifier with a custom classifier containing 10 output classes.
- Trained the newly added classifier using Adam optimizer and CrossEntropyLoss.

### Custom Classifier Architecture

| Layer | Configuration |
|---|---|
| Input | 25,088 features |
| Fully Connected | 1024 neurons |
| Activation | ReLU |
| Dropout | 0.5 |
| Fully Connected | 512 neurons |
| Activation | ReLU |
| Output | 10 classes |

### Training Configuration

- Pretrained Model: VGG16
- Weights: ImageNet
- Training Strategy: Feature Extraction
- Frozen Layers: Convolutional feature extractor
- Optimizer: Adam
- Loss Function: CrossEntropyLoss
- Training Epochs: 5

### Results

| Metric | Result |
|---|---:|
| Final Training Loss | 0.1834 |
| Final Training Accuracy | **93.67%** |
| Test Accuracy | Pending |

The pretrained convolutional layers remained frozen while the newly added classifier learned FashionMNIST-specific features.

## Model Comparison

| Metric | Baseline CNN | Optuna CNN | VGG16 |
|---|---:|---:|---:|
| Approach | CNN | Optimized CNN | Transfer Learning |
| Test Accuracy | 83.92% | Pending | Pending |
| Validation Accuracy | — | 92.15% | Pending |
| Training Accuracy | — | — | 93.67% |
| Hyperparameter Optimization | Manual | Optuna | Manual |
| Pretrained Weights | No | No | ImageNet |

**Note:** Validation accuracy, training accuracy, and test accuracy represent different evaluation metrics. A fair comparison requires evaluating all three models on the same untouched test dataset.

## Key Learnings

- Understanding CNN architecture and feature extraction.
- Working with PyTorch datasets and DataLoaders.
- Understanding training, validation, and testing workflows.
- Learning hyperparameter optimization using Optuna.
- Understanding how learning rate, dropout, filters, and hidden neurons affect performance.
- Understanding pretrained models and transfer learning.
- Freezing convolutional layers using `requires_grad`.
- Replacing pretrained classification heads for custom datasets.
- Comparing CNNs trained from scratch with pretrained architectures.

## Future Improvements

- Fine-tune the last convolutional block of VGG16.
- Experiment with different learning rates.
- Optimize batch size and image resolution.
- Compare VGG16 with ResNet18 and other pretrained architectures.
- Evaluate all models using the same test dataset.
- Compare training time, model size, and computational requirements.
