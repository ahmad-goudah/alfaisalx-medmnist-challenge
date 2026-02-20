# Task 3 — Semantic Image Retrieval System (PneumoniaMNIST)

## 1. Objective

The goal of this task is to build a semantic image retrieval system for the PneumoniaMNIST dataset.

Given a query image, the system retrieves the most visually and semantically similar images from the test set using learned feature embeddings.

---

## 2. Approach Overview

The retrieval system was implemented using the following pipeline:

1. Train a CNN classifier (Task 1).
2. Extract feature embeddings from the trained CNN.
3. Normalize embeddings for cosine similarity.
4. Index embeddings using FAISS.
5. Perform nearest-neighbor search.
6. Evaluate retrieval quality using Precision@k.

---

## 3. Embedding Extraction

- The trained CNN from Task 1 was used as a feature extractor.
- Embeddings were taken from the fully connected feature layer before the final classification layer.
- Embeddings were L2-normalized before indexing.
- Embedding dimension: 128

This ensures that retrieval is based on learned semantic representations rather than raw pixel similarity.

---

## 4. Indexing Method

Library used: **FAISS (Facebook AI Similarity Search)**

Index type:
- `IndexFlatIP` (inner product)
- Used with normalized vectors → equivalent to cosine similarity

All test set embeddings were added to the FAISS index.

---

## 5. Retrieval Process

For a given query image:

1. Compute its embedding using the trained CNN.
2. Normalize the embedding.
3. Search the FAISS index for top-k nearest neighbors.
4. Exclude the query image itself.
5. Return the top-k most similar images.

Example retrieval visualizations are saved in:

reports/task3/


---

## 6. Evaluation Metric

Retrieval performance was evaluated using:

- Precision@1
- Precision@5
- Precision@10

Precision@k is defined as:

Number of retrieved images with correct label / k

The results are stored in:

reports/task3/precision_at_k.txt


---

## 7. Example Outputs

Saved retrieval examples:

- reports/task3/retrieval_query_0.png
- reports/task3/retrieval_query_10.png
- reports/task3/retrieval_query_25.png

Each visualization shows:
- The query image
- The top-5 most similar retrieved images
- Their class labels

---

## 8. Observations

- Retrieval quality is strong when class separation is clear.
- Some errors occur due to:
  - Low image resolution (28×28)
  - Similar texture patterns between classes
  - Feature overlap in ambiguous cases

Overall, learned CNN embeddings provide meaningful semantic similarity for medical image retrieval.

---

## 9. How to Run

1. Run the main notebook:
notebooks/AlfaisalX_Challenge.ipynb


2. Execute all cells.

3. Retrieval outputs and evaluation results will be automatically saved into:


reports/task3/


---

## 10. Possible Improvements

If more time were available, the following improvements could be explored:

- Using a deeper backbone (e.g., ResNet18)
- Using contrastive or metric learning
- Using supervised contrastive loss
- Adding approximate indexing methods (IVF, HNSW)
- Adding text-to-image retrieval using multimodal models

