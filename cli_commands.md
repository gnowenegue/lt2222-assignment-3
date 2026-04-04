# Commands Available

## Part 2a — Train FastText model embeddings

```bash
python train_fasttext.py \
  --train dataset/train.tsv \
  --dev dataset/dev.tsv \
  --test dataset/test.tsv \
  --dimension 100 \
  --output model/fasttext.model \
  --window_size 5 \
  --min_count 1
```

## Part 2b — Generate embeddings

**Training set:**
```bash
python generate_embeddings.py \
  --input dataset/train.tsv \
  --model model/fasttext.model \
  --output embedding/train.npy
```

**Validation set:**
```bash
python generate_embeddings.py \
  --input dataset/dev.tsv \
  --model model/fasttext.model \
  --output embedding/dev.npy
```

**Test set:**
```bash
python generate_embeddings.py \
  --input dataset/test.tsv \
  --model model/fasttext.model \
  --output embedding/test.npy
```

## Part 3 — Neural topic classification

```bash
python train_classifier.py \
  --train dataset/train.tsv \
  --train_embeddings embedding/train.npy \
  --dev dataset/dev.tsv \
  --dev_embeddings embedding/dev.npy \
  --labels dataset/labels.txt \
  --epochs 100 \
  --batch_size 32 \
  --learning_rate 0.001 \
  --hidden_size 64 \
  --output model/classifier.pth \
  --plot report/evaluation_plot.png
```

## Part 4 — Evaluation

```bash
python evaluate_classifier.py \
  --input dataset/test.tsv \
  --embeddings embedding/test.npy \
  --model model/classifier.pth \
  --labels dataset/labels.txt \
  --hidden_size 64 \
  --batch_size 128 \
  --confusion_matrix report/confusion_matrix.png
```
