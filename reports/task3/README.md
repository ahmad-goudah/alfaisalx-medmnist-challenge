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
