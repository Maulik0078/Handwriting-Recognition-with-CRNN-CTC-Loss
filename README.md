# Named Entity Recognition: BiLSTM vs BERT

> CSC 483 – Applied Deep Learning  
> **Authors:** Asad Aziz Khan, Harshanbarath Rangaraju Suresh

A complete NER pipeline on the **CoNLL-2003** dataset comparing a from-scratch **BiLSTM** model against a fine-tuned **BERT** (`bert-base-cased`) model, implemented in Keras/TensorFlow and PyTorch respectively.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Results](#results)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Task Breakdown](#task-breakdown)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [License](#license)

---

## Overview

Named Entity Recognition (NER) is a sequence labeling task where each token in a sentence is assigned an entity tag (e.g., `B-PER`, `I-LOC`, `O`). This project implements and compares two architectures:

| Model | Framework | Approach |
|-------|-----------|----------|
| BiLSTM | TensorFlow / Keras | Trained from scratch |
| BERT | PyTorch (Hugging Face) | Fine-tuned from `bert-base-cased` |

---

## Results

| Model | Validation Micro F1 |
|-------|:-------------------:|
| BiLSTM (3 epochs) | ~0.794 |
| BERT (2 epochs) | ~0.941 |

BERT outperforms BiLSTM by ~15 F1 points thanks to large-scale pretraining and self-attention over the full sentence context.

---

## Project Structure

```
ner-bilstm-bert/
├── NER_BiLSTM_BERT.ipynb   # Main Jupyter notebook (all tasks)
├── requirements.txt         # Python dependencies
├── .gitignore
└── README.md
```

---

## Setup & Installation

### Prerequisites

- Python 3.8+
- pip

### Install dependencies

```bash
pip install -r requirements.txt
```

Or run the first cell in the notebook which installs everything automatically.

### GPU (optional but recommended for BERT)

If you have a CUDA-compatible GPU, PyTorch will use it automatically. Otherwise the code falls back to CPU.

---

## Usage

Open and run the notebook end-to-end:

```bash
jupyter notebook NER_BiLSTM_BERT.ipynb
```

Or open it in **Google Colab** for free GPU access:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<YOUR_USERNAME>/ner-bilstm-bert/blob/main/NER_BiLSTM_BERT.ipynb)

> Replace `<YOUR_USERNAME>` with your GitHub username after uploading.

---

## Task Breakdown

| Task | Description | Points |
|------|-------------|-------:|
| Task 1 | Dataset exploration — load CoNLL-2003, print split sizes, explain BIO tags | 15 |
| Task 2 | Data preprocessing — vocabulary building, integer encoding, padding | 15 |
| Task 3 | BiLSTM model — build, train (3 epochs), evaluate with seqeval F1 | 25 |
| Task 4 | BERT fine-tuning — subword tokenization, label alignment, training loop | 30 |
| Task 5 | Comparison & analysis — performance, self-attention, compute trade-offs | 15 |

---

## Model Architecture

### BiLSTM

```
Input (token IDs, max_len=128)
  └── Embedding (vocab_size × 64, mask_zero=True)
        └── Bidirectional LSTM (128 units, return_sequences=True)
              └── TimeDistributed Dense (9 classes, softmax)
```

### BERT Token Classifier

```
Input (subword IDs, max_len=128)
  └── bert-base-cased encoder (12 layers, 768 hidden, 12 heads)
        └── Linear classification head (768 → 9 classes)
```

Label alignment rules:
- `[CLS]` / `[SEP]` / `[PAD]` → `IGNORE_LABEL = -100`
- First subword of each word → original word-level NER label
- Continuation subwords → `IGNORE_LABEL`

---

## Dataset

**CoNLL-2003** English NER dataset (downloaded from a public mirror at runtime).

| Split | Sentences |
|-------|----------:|
| Train | ~14,041 |
| Validation | ~3,250 |
| Test | ~3,684 |

**NER label set:** `O`, `B-PER`, `I-PER`, `B-ORG`, `I-ORG`, `B-LOC`, `I-LOC`, `B-MISC`, `I-MISC`

BIO scheme:
- **B-** = beginning of an entity span
- **I-** = continuation of an entity span
- **O** = outside any entity

---

## License

This project is for academic purposes (CSC 483). Feel free to use the code for learning.
