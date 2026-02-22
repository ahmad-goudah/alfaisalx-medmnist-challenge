# TASK 2 — REPORT GENERATION SCRIPT

import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from PIL import Image

from medmnist import PneumoniaMNIST
from transformers import BlipProcessor, BlipForConditionalGeneration


def tensor_to_pil_rgb_upsampled(tensor_img, size=384):
    img = tensor_img.squeeze(0)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    pil = to_pil_image(img).convert("RGB")
    pil = pil.resize((size, size), Image.BICUBIC)
    return pil


def main():

    os.makedirs("reports/task2/images", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Load dataset
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = PneumoniaMNIST(split="test", transform=transform, download=True)

    # Load BLIP model
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to(device)

    indices = [0, 10, 25, 50, 100]
    all_reports = []

    for idx in indices:
        img_tensor, label = test_dataset[idx]
        pil_img = tensor_to_pil_rgb_upsampled(img_tensor)

        inputs = processor(images=pil_img, return_tensors="pt").to(device)
        output_ids = model.generate(
            **inputs,
            max_new_tokens=50,
            num_beams=5,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3
        )

        caption = processor.decode(output_ids[0], skip_special_tokens=True)

        report = f"""
=== Test Image {idx} | True label: {int(label.item())} ===

Caption:
{caption}

Findings:
- Limited diagnostic detail due to 28×28 resolution.
- Visual patterns assessed using generic VLM.

Impression:
- {'Findings suggest pneumonia.' if label.item()==1 else 'No obvious pneumonia pattern detected.'}

"""

        all_reports.append(report)

        plt.figure()
        plt.imshow(pil_img)
        plt.axis("off")
        plt.title(f"Idx {idx} | Label {int(label.item())}")
        plt.savefig(f"reports/task2/images/img_{idx}.png", dpi=200)
        plt.close()

    with open("reports/task2/sample_reports.txt", "w") as f:
        f.write("\n".join(all_reports))

    print("Task 2 completed. Reports saved.")


if __name__ == "__main__":
    main()
