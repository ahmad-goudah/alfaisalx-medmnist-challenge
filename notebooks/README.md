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

- reports/task1/training_loss.png
- reports/task1/accuracy vs epoch.png
- reports/task1/confusion_matrix.png
- reports/task1/roc_curve.png
- reports/task1/failure_cases.png
- reports/task1/val_acc_f1.png

  
---

##  Results Summary

The CNN achieved strong performance with validation ROC-AUC ≈ **0.99**, demonstrating effective class separability despite low image resolution.

### Failure Analysis

Observed errors include:

- Ambiguous opacity patterns  
- Borderline texture cases  
- Information bottleneck due to 28×28 resolution  

---

##  Key Insight

While performance is strong on PneumoniaMNIST, real-world deployment would require:

- Higher-resolution imaging  
- External validation  
- Clinical calibration  

---

#  Task 2 — Medical Report Generation

##  Objective

Generate structured radiology-style reports from chest X-ray images using Vision-Language Models (VLMs).

---

##  Approach Overview

Three strategies were evaluated:

---

### 1️⃣ General Vision-Language Model (BLIP)

**Model:** `Salesforce/blip-image-captioning-base`

- Generates generic image captions  
- Frequently produces non-medical descriptions  
- Demonstrates strong domain mismatch  

Example outputs:

- “a black and white photo of an empty road”
- “a person walking down a path”

---

### 2️⃣ Radiology-Tuned VLM (MIMIC-CXR Fine-Tuned BLIP)

**Model:** `nathansutton/generate-cxr`

- Fine-tuned for chest X-ray reports  
- Produced unstable or fragmented outputs on PneumoniaMNIST  

**Reason:**

- Extreme resolution mismatch (28×28 vs clinical X-rays)  
- Distribution shift in preprocessing  

---

### 3️⃣ Biomedical Language Model (BioGPT)

**Model:** `microsoft/BioGPT`

Two-stage pipeline:

1. BLIP generates an image caption  
2. BioGPT converts caption + clinical indication into structured report:
   - Findings  
   - Impression  

---

##  Observations

- General VLM suffers from strong domain shift  
- Radiology-tuned VLM struggles with low-resolution images  
- Biomedical LM improves medical phrasing but depends on meaningful visual features  

---

##  Scientific Insight

Effective medical report generation requires:

- Domain-aligned visual encoders  
- Sufficient image resolution  
- Strong multimodal alignment  

Language models alone cannot compensate for weak visual representations.

---

##  Outputs

- reports/task2/sample_reports.txt
- reports/task2/compare_medvlm_biogpt/comparison.md
- reports/task2/img_0.png
- reports/task2/img_10.png
- reports/task2/img_100.png
- reports/task2/img_25.png
- reports/task2/img_50.png


---

#  Task 3 — Semantic Image Retrieval

##  Objective

Build an image-to-image retrieval system using learned CNN embeddings.

---

##  Methodology

1. Extract 128-dimensional embeddings from CNN feature layer  
2. Apply L2 normalization  
3. Index embeddings using FAISS (`IndexFlatIP`)  
4. Perform cosine similarity search  
5. Evaluate using Precision@k  

---

##  Evaluation

**Metric:** Precision@k  

Results stored in:

- reports/task3/precision_at_k.txt

Example retrieval visualizations:

- reports/task3/retrieval_query_0.png
- reports/task3/retrieval_query_10.png
- reports/task3/retrieval_query_25.png

---

## Findings

- Supervised CNN embeddings transfer effectively to retrieval tasks  
- Class-consistent clustering observed  
- Retrieval performance validates representation quality  

---

# Limitations

- Extremely low resolution (28×28)  
- No domain-specific visual pretraining  
- No uncertainty calibration  
- Limited dataset diversity  

---

# Future Improvements

- Fine-tune pretrained medical CNN backbones  
- Use contrastive learning for improved embeddings  
- Apply radiology-trained multimodal models  
- Incorporate uncertainty estimation  
- Evaluate on higher-resolution datasets  

---

# Reproducibility

Main notebook:

- notebooks/AlfaisalX_Challenge.ipynb


### To reproduce:

1. Install dependencies  
2. Run notebook top-to-bottom  
3. All results and reports are generated automatically  

---

#  Author

**Dr.-Ing. Ahmad Abdullatif Goudah, Ph.D.**  
AI / Machine Learning Engineer  
University of Duisburg-Essen, Germany  
- ahmad.goudah@gmail.com
- +491783606461
- +201114558050
---

#  Challenge Completion Summary

| Task | Status |
|------|--------|
| Task 1 | ✔ CNN Classification with full evaluation |
| Task 2 | ✔ Multimodal Report Generation + Domain Analysis |
| Task 3 | ✔ Semantic Retrieval System |

All three tasks were completed with quantitative evaluation, qualitative analysis, and scientific interpretation.
