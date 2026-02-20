# Task 2 — Medical Report Generation (Vision-Language)

## 1) Dataset
- Dataset: MedMNIST v2 — PneumoniaMNIST
- Input images: 28×28 grayscale chest X-ray images
- Labels: 0 = Normal, 1 = Pneumonia
- We used the official test split for qualitative report generation.

## 2) Model used
- Vision-Language model: **Salesforce BLIP (image captioning base)**
- Reason: lightweight and stable in Colab, provides consistent image-text outputs for rapid prototyping.

## 3) Method
Because PneumoniaMNIST images are very low resolution (28×28), direct radiology-style prompting sometimes causes repetition (e.g., looping on “Impression”).
To ensure stable and interpretable outputs, we used a two-stage method:

1. **Stage A — Caption generation**
   - Generate a short, neutral description of the X-ray image using BLIP.
   - Use decoding constraints to reduce repetition (beam search, no_repeat_ngram_size, repetition_penalty).

2. **Stage B — Structured report formatting**
   - Convert the generated caption into a short structured medical-style report using a consistent template with:
     - Findings
     - Impression
   - Include a safety/quality note acknowledging low image resolution and recommending clinical correlation.

## 4) Outputs
- Generated example reports are saved in:
  - `reports/task2/sample_reports.txt`
- Example images used for generation are saved in:
  - `reports/task2/images/`

## 5) Observations
- The model produces generic visual descriptions due to low resolution.
- Structured formatting makes the output easier to interpret and aligns with clinical reporting style.
- Some cases remain ambiguous, and the report explicitly notes limitations.

## 6) Limitations
- The model is not specifically trained on radiology report datasets.
- PneumoniaMNIST resolution (28×28) limits clinical detail.
- Outputs should not be used for real medical decisions.

## 7) Improvements (if more time)
- Use a medical-specific VLM (e.g., radiology-trained model) and compare outputs.
- Fine-tune on radiology report datasets (e.g., MIMIC-CXR reports).
- Use higher resolution images and stronger calibration of uncertainty.
