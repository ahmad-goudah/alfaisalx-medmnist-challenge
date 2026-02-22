#  TASK 3 — RETRIEVAL SCRIPT

import os
import torch
import numpy as np
import faiss
import matplotlib.pyplot as plt

from torchvision import transforms
from torch.utils.data import DataLoader
from medmnist import PneumoniaMNIST


class SimpleCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, padding=1)
        self.pool = torch.nn.MaxPool2d(2,2)
        self.fc = torch.nn.Linear(64*7*7, 128)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)


def main():

    os.makedirs("reports/task3", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([transforms.ToTensor()])
    dataset = PneumoniaMNIST(split="test", transform=transform, download=True)
    loader = DataLoader(dataset, batch_size=256, shuffle=False)

    model = SimpleCNN().to(device)
    model.eval()

    embeddings = []
    labels = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            feat = model(x)
            embeddings.append(feat.cpu().numpy())
            labels.append(y.numpy())

    embeddings = np.vstack(embeddings).astype("float32")
    labels = np.vstack(labels).ravel()

    # Normalize embeddings
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(128)
    index.add(embeddings)

    # Retrieval example
    k = 5
    D, I = index.search(embeddings[:1], k)

    fig, axes = plt.subplots(1, k, figsize=(12,3))
    for j, idx in enumerate(I[0]):
        img, lab = dataset[idx]
        axes[j].imshow(img.squeeze(0), cmap="gray")
        axes[j].set_title(f"Label {int(lab)}")
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig("reports/task3/retrieval_example.png", dpi=200)
    plt.close()

    print("Task 3 completed. Retrieval example saved.")


if __name__ == "__main__":
    main()
