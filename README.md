# FashionMNIST Image Classification with PyTorch & Optuna

A deep learning project that classifies FashionMNIST images using Convolutional Neural Networks (CNNs) built with PyTorch. This project explores the impact of hyperparameter tuning using **Optuna** to improve model performance.

## Project Overview

The project implements two CNN approaches:

1. **Baseline CNN:** A CNN trained using manually selected hyperparameters.
2. **Optimized CNN:** A CNN whose hyperparameters are optimized using Optuna to achieve better validation performance.

The project demonstrates the complete deep learning workflow, including data preprocessing, model building, training, evaluation, and hyperparameter optimization.

## Features

- CNN implemented using PyTorch
- FashionMNIST dataset from Torchvision
- Image preprocessing using Transforms
- DataLoader for batch processing
- Model training and evaluation
- Hyperparameter optimization using Optuna
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

## Model 1: Baseline CNN

The first model uses manually selected hyperparameters and a CNN architecture with convolutional, pooling, and fully connected layers.

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

## Model Comparison

| Metric | Baseline CNN | Optuna-Tuned CNN |
|---|---:|---:|
| Hyperparameter Selection | Manual | Optuna |
| Test Accuracy | 83.92% | Pending |
| Validation Accuracy | — | **92.15%** |
| Optimization Trials | — | 20 |

**Note:** The baseline test accuracy and optimized validation accuracy are measured on different data splits. A fair final comparison requires evaluating the optimized model on the same untouched test dataset.

## Key Learnings

- Understanding CNN architecture and feature extraction.
- Working with PyTorch datasets and DataLoaders.
- Understanding training and validation workflows.
- Learning hyperparameter optimization with Optuna.
- Understanding how learning rate, dropout, filters, and hidden neurons affect model performance.
- Comparing manually configured and optimized models.

## Future Improvements

- Evaluate the optimized model on the test dataset.
- Experiment with learning-rate schedulers.
- Apply data augmentation.
- Explore deeper
