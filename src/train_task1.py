# Training script with configurable hyperparameters

import os
import json
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import transforms

from medmnist import PneumoniaMNIST


# -------------------------
# Utilities
# -------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_mean_std(dataset, max_batches: int = 50, batch_size: int = 256) -> tuple[float, float]:
    """
    Compute dataset mean/std for medical image normalization.
    PneumoniaMNIST is small and grayscale, so mean/std is stable.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    means = []
    stds = []
    count = 0
    for x, _ in loader:
        # x: [B,1,28,28]
        means.append(x.mean().item())
        stds.append(x.std().item())
        count += 1
        if count >= max_batches:
            break
    return float(np.mean(means)), float(np.mean(stds))


# -------------------------
# Model (small CNN, justified for 28x28 + CPU)
# -------------------------
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
        x = self.pool(F.relu(self.conv1(x)))  # 28 -> 14
        x = self.pool(F.relu(self.conv2(x)))  # 14 -> 7
        x = self.pool(F.relu(self.conv3(x)))  # 7 -> 3
        x = x.view(x.size(0), -1)
        x = self.drop(x)
        x = F.relu(self.fc1(x))
        logit = self.fc2(x)
        return logit


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device).float()  # [B,1]
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_loss(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device).float()
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser()
    # Hyperparameters (configurable)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)

    # Scheduler
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["none", "step", "cosine"])
    parser.add_argument("--step_size", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=0.5)

    # Paths
    parser.add_argument("--model_out", type=str, default="models/task1_best.pt")
    parser.add_argument("--history_out", type=str, default="reports/task1/train_history.json")
    parser.add_argument("--use_augmentation", action="store_true")

    args = parser.parse_args()

    set_seed(args.seed)

    os.makedirs("models", exist_ok=True)
    os.makedirs("reports/task1", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Basic dataset (temporary, to compute mean/std)
    base_transform = transforms.Compose([transforms.ToTensor()])
    train_base = PneumoniaMNIST(split="train", transform=base_transform, download=True)

    mean, std = compute_mean_std(train_base)
    if std < 1e-6:
        std = 1.0
    print(f"Computed mean={mean:.4f}, std={std:.4f}")

    # Medical-image meaningful augmentation for CXR (light + safe)
    # (Avoid horizontal flip for CXR unless justified; small rotations/translation are common)
    if args.use_augmentation:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.Normalize(mean=[mean], std=[std]),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[mean], std=[std]),
        ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[mean], std=[std]),
    ])

    # Proper split handling (official MedMNIST splits)
    train_ds = PneumoniaMNIST(split="train", transform=train_transform, download=True)
    val_ds = PneumoniaMNIST(split="val", transform=test_transform, download=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = SmallCNN(dropout=args.dropout).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # LR scheduling (required by spec)
    if args.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = None

    history = {"train_loss": [], "val_loss": [], "lr": [], "mean": mean, "std": std, "args": vars(args)}
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = eval_loss(model, val_loader, criterion, device)

        lr_now = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(lr_now)

        print(f"Epoch {epoch:02d}/{args.epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | lr={lr_now:.6f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.model_out)
            print("  ✅ Saved best model:", args.model_out)

        if scheduler is not None:
            scheduler.step()

    # Save training history/config
    with open(args.history_out, "w") as f:
        json.dump(history, f, indent=2)
    print("Saved training history:", args.history_out)


if __name__ == "__main__":
    main()
