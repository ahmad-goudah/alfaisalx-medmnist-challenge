# task2_report_generation

  ## Domain Adaptation Analysis 
  
Across experiments with three different multimodal strategies:

1. General-purpose VLM (BLIP)
2. Radiology-tuned VLM (MIMIC-CXR fine-tuned BLIP)
3. Biomedical language model (BioGPT) combined with visual captions

we observed consistent limitations when applied to PneumoniaMNIST.

The general BLIP model produced non-medical captions (e.g., "empty road", "person walking"), indicating strong domain mismatch between natural image-caption datasets and chest radiographs.

The radiology-tuned BLIP model generated unstable or fragmented outputs when applied to 28×28 images, suggesting that fine-tuning alone does not compensate for extreme resolution and preprocessing differences.

The biomedical language model was unable to reliably expand generic visual captions into clinically meaningful reports, demonstrating that structured medical language modeling depends on semantically relevant upstream visual representations.

These findings highlight an important principle:

Effective medical report generation requires both domain-aligned visual encoders and sufficiently informative image resolution. When either component is misaligned, multimodal generation quality degrades significantly.

This experiment demonstrates the challenges of transferring foundation models to low-resolution derived medical datasets.
