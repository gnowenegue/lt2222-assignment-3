#!/bin/bash

# exit on error, treat unset variables as errors and catch pipe failures
set -euo pipefail

echo "===== Part 2a — Sentence embeddings (Step 1: Training FastText) ====="
python train_fasttext.py \
  --train dataset/train.tsv \
  --dev dataset/dev.tsv \
  --test dataset/test.tsv \
  --dimension 100 \
  --output model/fasttext.model

echo -e "\n===== Part 2b — Sentence embeddings (Step 2: Generating Vectors) ====="
python generate_embeddings.py --input dataset/train.tsv --output embedding/train.npy
python generate_embeddings.py --input dataset/dev.tsv --output embedding/dev.npy
python generate_embeddings.py --input dataset/test.tsv --output embedding/test.npy

echo -e "\n===== Part 3 — Neural topic classification ====="
python train_classifier.py \
  --train dataset/train.tsv \
  --train_embeddings embedding/train.npy \
  --dev dataset/dev.tsv \
  --dev_embeddings embedding/dev.npy \
  --epochs 100 \
  --plot report/evaluation_plot.png

echo -e "\n===== Part 4 — Evaluation ====="
python evaluate_classifier.py \
  --input dataset/test.tsv \
  --embeddings embedding/test.npy \
  --confusion_matrix report/confusion_matrix.png

echo -e "\n===== PIPELINE COMPLETE! ====="
