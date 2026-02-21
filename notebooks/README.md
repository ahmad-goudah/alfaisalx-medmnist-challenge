#  AlfaisalX Postdoctoral Technical Challenge  
## MedMNIST v2 – PneumoniaMNIST  
**Research Fellow / Postdoctoral Position**  
Cognitive Robotics & Autonomous Agents – MedX Research Unit  

---

#  Project Overview

This repository contains my complete implementation of the 7-Day Postdoctoral Technical Challenge using the **PneumoniaMNIST** dataset from MedMNIST v2.

The project includes:

-  **Task 1 — CNN Classification with Comprehensive Analysis**
-  **Task 2 — Medical Report Generation using Vision-Language Models**
-  **Task 3 — Semantic Image Retrieval System**

All experiments are reproducible using the provided Google Colab notebook.

---

#  Dataset

**Dataset:** PneumoniaMNIST (MedMNIST v2)

- Binary classification task  
- Image size: **28×28 grayscale**
- Classes:
  - `0` → Normal  
  - `1` → Pneumonia  

### Official Splits

| Split      | Size |
|------------|------|
| Train      | 4708 |
| Validation | 524  |
| Test       | 624  |

The dataset is lightweight and suitable for reproducible experimentation on standard hardware.

---

# Task 1 — CNN Classification

## Objective

Develop a deep learning classifier for pneumonia detection with full quantitative evaluation and qualitative failure analysis.

---

## Model Architecture

A lightweight Convolutional Neural Network (CNN) was implemented:

- Conv2D → ReLU → MaxPool  
- Conv2D → ReLU → MaxPool  
- Conv2D → ReLU → MaxPool  
- Fully Connected Feature Layer  
- Dropout (regularization)  
- Final Linear Layer (binary output)

### Training Setup

- **Loss:** BCEWithLogitsLoss  
- **Optimizer:** Adam  
- **Learning rate:** 1e-3  
- **Batch size:** 128  
- **Epochs:** 15  
- **Model selection:** Best validation ROC-AUC  

---

## Evaluation Metrics

- Accuracy  
- Precision  
- Recall  
- F1-Score  
- ROC-AUC  
- Confusion Matrix  

### Generated Plots
