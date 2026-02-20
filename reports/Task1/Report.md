Task 1 — CNN Classification (PneumoniaMNIST)
1) Dataset & Preprocessing

The dataset used is MedMNIST v2 – PneumoniaMNIST, a binary medical image classification dataset derived from chest X-ray images.

Task type: Binary classification

Classes:

0 → Normal

1 → Pneumonia

Image size: 28 × 28 (grayscale)

Official splits were used:

Training set: 4708 images

Validation set: 524 images

Test set: 624 images

Preprocessing

Images were converted to PyTorch tensors.

Pixel values were scaled to the range [0, 1].

Light data augmentation was applied only to the training set:

Random rotation (±10 degrees)

Random resized crop (scale between 0.9 and 1.0)

Validation and test sets were not augmented.

This setup ensures better generalization while preserving evaluation fairness.

2) Model Architecture

A lightweight Convolutional Neural Network (CNN) was designed to efficiently handle 28×28 grayscale images.

Architecture:

Conv2D (1 → 32), kernel=3, padding=1

ReLU

MaxPooling (2×2)

Conv2D (32 → 64), kernel=3, padding=1

ReLU

MaxPooling (2×2)

Conv2D (64 → 128), kernel=3, padding=1

ReLU

MaxPooling (2×2)

Flatten

Fully Connected (128 → 128)

Dropout (0.25)

Output layer (128 → 1 logit)

The model outputs a single logit for binary classification, followed by a sigmoid during evaluation.

This architecture is computationally efficient and well-suited for low-resolution medical images.

3) Training Setup

Loss function: BCEWithLogitsLoss

Optimizer: Adam

Learning rate: 0.001

Batch size: 128

Number of epochs: 15

Best model selection: Based on highest validation AUC

Training and validation losses were tracked across epochs, and the best checkpoint was saved automatically.

4) Results

Evaluation was performed on the official test set using the best saved model.

Test Metrics
