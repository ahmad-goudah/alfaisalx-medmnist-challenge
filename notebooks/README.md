🧠 AlfaisalX Postdoctoral Technical Challenge
MedMNIST v2 – PneumoniaMNIST

Research Fellow / Postdoctoral Position
Cognitive Robotics & Autonomous Agents – MedX Research Unit

📌 Project Overview

This repository contains my full implementation of the 7-Day Postdoctoral Technical Challenge using the PneumoniaMNIST dataset from MedMNIST v2.

The challenge required building end-to-end AI systems for medical imaging, including:

✅ Task 1 — CNN Classification with Comprehensive Analysis

✅ Task 2 — Medical Report Generation using Vision-Language Models

✅ Task 3 — Semantic Image Retrieval System

All experiments are fully reproducible via the provided Google Colab notebook.

📂 Dataset

Dataset: PneumoniaMNIST (MedMNIST v2)

Binary classification task

Image size: 28×28 grayscale

Classes:

0 → Normal

1 → Pneumonia

Official splits:

Split	Size
Train	4708
Val	524
Test	624

The dataset is intentionally lightweight to allow reproducible experimentation on standard hardware.

✅ Task 1 — CNN Classification
🎯 Objective

Develop a deep learning classifier for pneumonia detection with full evaluation, visualization, and failure analysis.

🏗 Model Architecture

A lightweight CNN was implemented:

Conv2D → ReLU → MaxPool

Conv2D → ReLU → MaxPool

Conv2D → ReLU → MaxPool

Fully Connected (Feature Layer)

Dropout (regularization)

Final Linear Layer (binary output)

Loss function: BCEWithLogitsLoss
Optimizer: Adam
Learning rate: 1e-3
Batch size: 128
Epochs: 15

Best model selected using validation AUC.

📊 Evaluation Metrics

Accuracy

Precision

Recall

F1-score

ROC-AUC

Confusion Matrix

Plots generated:

reports/task1/training_loss.png
reports/task1/training_accuracy.png
reports/task1/confusion_matrix.png
reports/task1/roc_curve.png
reports/task1/failure_cases.png
📈 Results Summary

The model achieved strong performance with high ROC-AUC (~0.99 on validation), indicating strong class separability despite low image resolution.

Failure case analysis revealed:

Ambiguous texture patterns

Low-resolution information bottleneck

Mild opacity cases near class boundary

🔎 Key Insight

Even simple CNNs can perform well on PneumoniaMNIST due to dataset characteristics, but real-world clinical transfer would require higher-resolution imaging and external validation.

✅ Task 2 — Medical Report Generation
🎯 Objective

Generate structured radiology-style reports from chest X-ray images using Vision-Language Models (VLMs).

🧠 Approach

A multi-strategy evaluation was conducted:

1️⃣ General Vision-Language Model (BLIP)

Model: Salesforce/blip-image-captioning-base

Generates generic image captions.

Outputs often reflect domain mismatch (natural image bias).

Example captions:

“a black and white photo of an empty road”

“a person walking down a path”

This demonstrates strong domain shift between natural image-caption datasets and medical radiographs.

2️⃣ Radiology-Tuned VLM (MIMIC-CXR Fine-Tuned BLIP)

Model: nathansutton/generate-cxr

Designed for chest X-ray report generation.

However, unstable outputs were observed on PneumoniaMNIST.

Reason:

PneumoniaMNIST images are 28×28 grayscale.

Severe resolution mismatch compared to real clinical X-rays.

Domain shift remains significant.

3️⃣ Biomedical Language Model (BioGPT)

Model: microsoft/BioGPT

Used as a second-stage medical text generator:

BLIP generates image caption.

BioGPT converts caption + clinical indication into structured report:

Findings

Impression

📊 Observations

General BLIP produces non-medical captions (domain mismatch).

Radiology-tuned BLIP struggles with low-resolution inputs.

Biomedical LM improves clinical phrasing but depends on meaningful visual semantics.

🔬 Key Scientific Insight

Effective medical report generation requires:

Domain-aligned visual encoders

Adequate image resolution

Multimodal alignment

When upstream visual features are weak, language models cannot compensate.

This experiment highlights:

Distribution shift challenges

Resolution bottlenecks

Limitations of foundation model transfer

📁 Outputs
reports/task2/sample_reports.txt
reports/task2/compare_medvlm_biogpt/comparison.txt
reports/task2/images/
✅ Task 3 — Semantic Image Retrieval
🎯 Objective

Build an image-to-image retrieval system based on learned CNN embeddings.

🧠 Method

Extract 128-dimensional embeddings from trained CNN feature layer.

Normalize embeddings (L2 normalization).

Index embeddings using FAISS.

Perform cosine similarity search.

Evaluate using Precision@k.

FAISS index used:

IndexFlatIP (Inner Product)

Cosine similarity via normalized embeddings

📊 Evaluation

Metric: Precision@k

Results stored in:

reports/task3/precision_at_k.txt

Example retrieval visualizations:

reports/task3/retrieval_query_*.png
🔎 Findings

Supervised CNN embeddings transfer effectively to similarity search.

Class-consistent clustering observed.

Retrieval performance supports representation quality.

⚠ Limitations

Extremely low resolution (28×28)

No medical-specific visual pretraining

No uncertainty calibration

Limited dataset diversity

🚀 Future Improvements

Fine-tune pretrained medical backbones (e.g., ResNet on ChestX-ray datasets)

Apply contrastive learning for better embeddings

Use radiology-trained multimodal models

Incorporate uncertainty estimation

Evaluate on higher-resolution datasets

🔁 Reproducibility

Main notebook:

notebooks/AlfaisalX_Challenge.ipynb

To reproduce:

Install dependencies

Run notebook top-to-bottom

All reports and plots will be generated automatically

👤 Author

Dr.-Ing. Ahmad Abdullatif Goudah, Ph.D.
AI / Machine Learning Engineer
University of Duisburg-Essen

✅ Challenge Completion Summary
Task	Status
Task 1	✔ CNN Classification with full evaluation
Task 2	✔ Multimodal report generation + domain analysis
Task 3	✔ Semantic retrieval with FAISS
