# Comparative Sentence Identification

This directory contains a standalone binary sentence-classification task. It reads the original COQE train/dev/test files but does not load tuple labels or build CRF extraction heads.

Run from the repository root:

```bash
CUDA_VISIBLE_DEVICES=0 python -m tasks.csi.train \
  --data_dir data/Ele-COQE \
  --model_type bert \
  --model_name_or_path /root/autodl-tmp/pretrain_model/base_chinese \
  --output_dir /root/autodl-tmp/outputs/csi/Ele-COQE/bert \
  --epochs 1 \
  --batch_size 16
```

Run the standalone LSTM baseline with the architecture defaults used by the legacy COQE code:

```bash
CUDA_VISIBLE_DEVICES=0 python -m tasks.csi.train \
  --data_dir data/Ele-COQE \
  --model_type lstm \
  --output_dir /root/autodl-tmp/outputs/csi/Ele-COQE/lstm \
  --epochs 25 \
  --batch_size 16
```

The LSTM defaults to 300-dimensional character embeddings, a two-layer bidirectional LSTM with hidden size 512, seed 2021, encoder learning rate `1e-4`, and classifier learning rate `2e-5`. Padding is excluded through packed sequences and hidden states start deterministically at zero.

The output directory contains the best checkpoint, run configuration, per-epoch history, dev/test metrics, tokenizer files, and sentence-level prediction TSV files.

Reported metrics are accuracy, positive-class precision/recall/F1, macro-F1, and the confusion matrix. Accuracy is retained for comparison with the original COQE paper.
