# Handwriting Recognition with CRNN + CTC Loss

> CSC 483 – Applied Deep Learning

A full handwriting recognition pipeline that combines **CNNs**, **Bidirectional LSTMs**, and **CTC loss** to recognize handwritten names from images — no pre-aligned labels required.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Results](#results)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Key Concepts](#key-concepts)
- [License](#license)

---

## Overview

Handwritten text varies wildly in style, spacing, slant, and noise. Unlike classification tasks, the input image and target text sequence are **not aligned** — a standard cross-entropy loss cannot handle this.

This project solves that with a **CRNN** (CNN + RNN) architecture trained with **CTC (Connectionist Temporal Classification) loss**, which learns to map image sequences to text without needing per-pixel or per-timestep annotations.

---

## Results

| Metric | Value (10 epochs) |
|--------|:-----------------:|
| Training Loss | tracked via `model.fit` |
| Validation Loss | tracked via `model.fit` |
| Mean Edit Distance | tracked per epoch via callback |

> Predictions are visualized with **green titles** for correct matches and **red titles** for mismatches on the test set.

---

## Project Structure

```
handwriting-recognition/
├── Handwriting_Recognition.ipynb   # Main notebook (all tasks)
├── requirements.txt                 # Python dependencies
├── .gitignore
└── README.md
```

---

## Setup & Installation

### Prerequisites

- Python 3.8+
- A **GPU runtime** is strongly recommended (e.g., Google Colab T4)

### Install dependencies

```bash
pip install -r requirements.txt
```

Or let the notebook's first cell handle it automatically.

### Dataset

The dataset (~400k handwritten name images) is downloaded automatically from Google Drive inside the notebook using `gdown`. No manual download needed.

---

## Usage

### Google Colab (recommended)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<YOUR_USERNAME>/handwriting-recognition/blob/main/Handwriting_Recognition.ipynb)

> Replace `<YOUR_USERNAME>` with your GitHub handle after uploading.

Make sure to switch to a **GPU runtime**: `Runtime → Change runtime type → T4 GPU`

### Local

```bash
jupyter notebook Handwriting_Recognition.ipynb
```
---

## Model Architecture

```
Input image  (256 × 64 × 1)
  └── Conv2D(32, 3×3, relu) + MaxPool(2×2)       →  128 × 32 × 32
        └── Conv2D(64, 3×3, relu) + MaxPool(2×2) →   64 × 16 × 64
              └── Reshape → (64, 16 × 64 = 1024)  [sequence of 64 timesteps]
                    └── Dense(64, relu) + Dropout(0.2)
                          └── BiLSTM(128, return_sequences=True)
                                └── BiLSTM(64, return_sequences=True)
                                      └── Dense(vocab_size + 2, softmax)
                                            └── CTCLayer (loss only at train time)
```

**Why `vocab_size + 2`?**  
CTC reserves two extra tokens internally: a **blank token** (used to separate repeated characters) and a catch-all for unknowns. The +2 accounts for both.

---

## Dataset

| Property | Value |
|----------|-------|
| Source | [Kaggle — Handwritten Names](https://www.kaggle.com/) (via Google Drive mirror) |
| Total images | ~400,000 |
| Used for training | 100,000 (configurable) |
| Format | PNG, grayscale |
| Labels | CSV mapping filename → name string |

**Splits used:**

| Split | Size |
|-------|-----:|
| Train | 90,000 |
| Validation | 5,000 |
| Test | 5,000 |

---

## Key Concepts

### Distortion-Free Resizing
All images are resized **with padding** (`preserve_aspect_ratio=True`) rather than forced stretching. This preserves the natural proportions of handwritten letters — crucial for recognition accuracy.

### CTC Loss
CTC allows the model to predict variable-length sequences from fixed-length feature maps without needing aligned labels. It sums over all valid alignments between the prediction and the target sequence, and `input_length` / `label_length` are passed explicitly so it knows where real content ends vs. padding.

### Edit Distance Callback
A custom `EditDistanceCallback` computes the **mean character edit distance** between predicted and ground-truth strings at the end of every epoch, giving an intuitive real-world accuracy signal beyond loss values.

---

## License

This project is for academic purposes (CSC 483). Feel free to use the code for learning.
