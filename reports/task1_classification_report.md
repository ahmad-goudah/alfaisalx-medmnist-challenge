 # Task 1 — CNN Classification Report (PneumoniaMNIST)

## Dataset & preprocessing

**Dataset:** MedMNIST v2 — PneumoniaMNIST  
**Task:** Binary classification (0 = Normal, 1 = Pneumonia)  
**Image format:** 28×28 grayscale

**Split handling (official):**
- Train split used for training
- Validation split used for model selection
- Test split used only for final evaluation

### Normalization (medical-image appropriate)
To stabilize training and support generalization, the training split statistics were computed and used for normalization:
- Mean and std computed from the training split (saved in `reports/task1/train_history.json`)
- Applied `Normalize(mean, std)` to train/val/test consistently

### Data augmentation (meaningful for chest X-rays)
Augmentations were limited to medically plausible transformations:
- Small rotation (±10°)
- Small translation (≤5%)
- Slight scale change (0.95–1.05)
- Mild brightness/contrast jitter

No aggressive transforms were used that might distort clinical meaning.

### Batch processing
Training and evaluation are performed using PyTorch `DataLoader` with batch processing for efficiency.

---

## Model architecture (layers + justification)

A lightweight CNN was implemented (`SmallCNN`) designed for 28×28 inputs and CPU-friendly training:

- Conv(1→32, 3×3) + ReLU + MaxPool
- Conv(32→64, 3×3) + ReLU + MaxPool
- Conv(64→128, 3×3) + ReLU + MaxPool
- Flatten
- FC feature layer (128 units) + Dropout
- FC output layer (1 logit)

**Justification:** PneumoniaMNIST images are small (28×28). A compact CNN is sufficient to capture relevant texture/opacity patterns without excessive capacity.

---

## Training methodology & hyperparameters

Training script: `src/train_task1.py` (configurable via CLI)

- Loss: `BCEWithLogitsLoss`
- Optimizer: Adam
- Learning rate: 1e-3
- Batch size: 128
- Epochs: 15
- LR scheduling: CosineAnnealingLR (or StepLR, configurable)
- Model selection: best validation loss (saved checkpoint)

Saved artifacts:
- Model weights: `models/task1_best.pt`
- Training history/config: `reports/task1/train_history.json`

---

## Evaluation metrics + visualizations

Evaluation script: `src/eval_task1.py`

Metrics reported on test split:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

Confusion matrix + ROC curve are generated for error analysis.



Example:
- Accuracy: 0.8814102564102564
- Precision: 0.8542600896860987
- Recall: 0.9769230769230769
- F1: 0.9114832535885168
- ROC-AUC: 0.9534845496383958
Confusion Matrix:
[[169  65]
 [  9 381]]


### Generated visualizations
Saved to `reports/task1/`:
- `training_loss.png`
- `confusion_matrix.png`
- `roc_curve.png`
- `failure_cases.png`

---

## Failure case analysis

Misclassified examples are visualized in:
- `reports/task1/failure_cases.png`

**Observed failure patterns (typical causes):**
- Low-resolution ambiguity: 28×28 limits fine-grained lung details
- Overlapping texture patterns between classes
- Borderline cases where opacity is subtle

These insights highlight where higher-resolution inputs or stronger backbones could help.

---

## Strengths, limitations, and improvements

### Strengths
- Strong performance and high ROC-AUC on official test split
- Lightweight model and fast training
- Complete reproducible pipeline with evaluation + visual diagnostics

### Limitations
- Very low image resolution limits clinical detail
- Simple CNN may miss higher-level semantic cues
- No external dataset validation

### Improvements (if more time)
- Use a pretrained backbone (e.g., ResNet18) with upsampling
- Try class-balanced loss or calibration
- Stronger but safe augmentations (noise, histogram adjustments)
- External validation on higher-resolution CXR datasets
