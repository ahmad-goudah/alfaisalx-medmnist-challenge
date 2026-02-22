# This script outputs:
# accuracy, precision, recall, F1, AUC
# confusion matrix
# ROC curve
# training loss curve
# failure_cases.png

import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torchvision import transforms
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)

from medmnist import PneumoniaMNIST


class SmallCNN(nn.Module):
    def __init__(self, dropout: float = 0.25):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout(dropout)
        self.fc1 = nn.Linear(128 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.drop(x)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs_all, preds_all, y_all = [], [], []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device).float()
        logits = model(x)
        probs = torch.sigmoid(logits).cpu().numpy().ravel()
        preds = (probs >= 0.5).astype(int)
        probs_all.append(probs)
        preds_all.append(preds)
        y_all.append(y.cpu().numpy().ravel().astype(int))
    return np.concatenate(y_all), np.concatenate(preds_all), np.concatenate(probs_all)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="models/task1_best.pt")
    parser.add_argument("--history_path", type=str, default="reports/task1/train_history.json")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--out_dir", type=str, default="reports/task1")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Load mean/std from training history (reproducible normalization)
    with open(args.history_path, "r") as f:
        hist = json.load(f)
    mean = hist.get("mean", 0.5)
    std = hist.get("std", 0.2)
    if std < 1e-6:
        std = 1.0

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[mean], std=[std]),
    ])

    test_ds = PneumoniaMNIST(split="test", transform=transform, download=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = SmallCNN(dropout=hist.get("args", {}).get("dropout", 0.25)).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    y_true, y_pred, y_prob = predict(model, test_loader, device)

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob)

    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(auc),
        "confusion_matrix": cm.tolist(),
        "mean": float(mean),
        "std": float(std),
    }

    with open(os.path.join(args.out_dir, "test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("Saved metrics:", os.path.join(args.out_dir, "test_metrics.json"))
    print(metrics)

    # --- Plots: training curves (from history)
    train_loss = hist["train_loss"]
    val_loss = hist["val_loss"]

    plt.figure()
    plt.plot(train_loss, label="train_loss")
    plt.plot(val_loss, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "training_loss.png"), dpi=200)
    plt.close()

    # Simple accuracy curve (recompute on val is possible; for quick compliance we plot losses + optionally add training_accuracy if tracked)
    # If you want real accuracy curves, track them in training history (recommended).

    # --- Confusion matrix
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "confusion_matrix.png"), dpi=200)
    plt.close()

    # --- ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "roc_curve.png"), dpi=200)
    plt.close()

    # --- Failure cases (misclassified)
    mis_idx = np.where(y_true != y_pred)[0]
    print("Misclassified:", len(mis_idx))

    # pick top 12 most confident wrong predictions
    conf = np.where(y_pred == 1, y_prob, 1 - y_prob)
    top = mis_idx[np.argsort(-conf[mis_idx])][:12]

    # plot images
    plt.figure(figsize=(10, 6))
    for k, idx in enumerate(top):
        img, lab = test_ds[int(idx)]  # tensor normalized; for visualization undo normalization approx
        img = img * std + mean
        img_np = img.squeeze(0).numpy()

        plt.subplot(3, 4, k + 1)
        plt.imshow(img_np, cmap="gray")
        plt.axis("off")
        plt.title(f"T={int(y_true[idx])} P={int(y_pred[idx])}\nprob={y_prob[idx]:.2f}")

    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "failure_cases.png"), dpi=200)
    plt.close()

    print("Saved plots to:", args.out_dir)


if __name__ == "__main__":
    main()
