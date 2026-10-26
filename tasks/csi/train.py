import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from tasks.csi.data import (
    BertBatchCollator,
    LstmBatchCollator,
    SentenceDataset,
    build_char_vocab,
    read_coqe_split,
)
from tasks.csi.metrics import classification_metrics
from tasks.csi.model import BertSentenceClassifier, LstmSentenceClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone comparative sentence identification")
    parser.add_argument("--data_dir", required=True, help="Directory containing train.txt, dev.txt and test.txt")
    parser.add_argument("--model_type", choices=("bert", "lstm"), default="bert")
    parser.add_argument("--model_name_or_path", help="Required for BERT: Hugging Face model id or local directory")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--lstm_learning_rate", type=float, default=1e-4)
    parser.add_argument("--lstm_classifier_learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--embedding_dim", type=int, default=300)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--selection_metric", choices=("accuracy", "f1", "macro_f1"), default="accuracy")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested but CUDA is unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_loader(examples, collator, batch_size: int, shuffle: bool, num_workers: int):
    return DataLoader(
        SentenceDataset(examples),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
    )


def train_epoch(model, loader, optimizer, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0
    for encoded, labels, _ in tqdm(loader, desc="train", leave=False):
        encoded = {key: value.to(device) for key, value in encoded.items()}
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(**encoded)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        total_examples += labels.size(0)
    return total_loss / total_examples


@torch.no_grad()
def evaluate(model, loader, device: torch.device) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    model.eval()
    losses: List[float] = []
    gold: List[int] = []
    predicted: List[int] = []
    rows: List[Dict[str, object]] = []

    for encoded, labels, examples in tqdm(loader, desc="evaluate", leave=False):
        encoded = {key: value.to(device) for key, value in encoded.items()}
        labels = labels.to(device)
        logits = model(**encoded)
        losses.append(F.cross_entropy(logits, labels, reduction="sum").item())
        probabilities = torch.softmax(logits, dim=-1)[:, 1].cpu().tolist()
        batch_predictions = logits.argmax(dim=-1).cpu().tolist()
        batch_gold = labels.cpu().tolist()

        gold.extend(batch_gold)
        predicted.extend(batch_predictions)
        for example, prediction, probability in zip(examples, batch_predictions, probabilities):
            rows.append(
                {
                    "sample_id": example.sample_id,
                    "gold_label": example.label,
                    "predicted_label": prediction,
                    "positive_probability": probability,
                    "text": example.text,
                }
            )

    metrics = classification_metrics(gold, predicted)
    metrics["loss"] = sum(losses) / len(gold)
    return metrics, rows


def save_json(path: Path, value: Dict[str, object]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def save_predictions(path: Path, rows: List[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("sample_id", "gold_label", "predicted_label", "positive_probability", "text"),
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.model_type == "bert" and not args.model_name_or_path:
        raise ValueError("--model_name_or_path is required when --model_type bert")
    set_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir)
    splits = {
        name: read_coqe_split(data_dir / f"{name}.txt")
        for name in ("train", "dev", "test")
    }
    print({name: len(examples) for name, examples in splits.items()})
    print(f"device={device}")

    tokenizer = None
    vocab = None
    if args.model_type == "bert":
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
        collator = BertBatchCollator(tokenizer, args.max_length)
    else:
        vocab = build_char_vocab(splits["train"])
        collator = LstmBatchCollator(vocab, args.max_length)
        print(f"vocab_size={len(vocab)}")

    loaders = {
        "train": make_loader(
            splits["train"], collator, args.batch_size, True, args.num_workers
        ),
        "dev": make_loader(
            splits["dev"], collator, args.batch_size, False, args.num_workers
        ),
        "test": make_loader(
            splits["test"], collator, args.batch_size, False, args.num_workers
        ),
    }

    if args.model_type == "bert":
        model = BertSentenceClassifier(args.model_name_or_path, dropout=args.dropout).to(device)
        optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        model = LstmSentenceClassifier(
            vocab_size=len(vocab),
            embedding_dim=args.embedding_dim,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
        ).to(device)
        optimizer = AdamW(
            [
                {
                    "params": list(model.embedding.parameters()) + list(model.encoder.parameters()),
                    "lr": args.lstm_learning_rate,
                },
                {"params": model.classifier.parameters(), "lr": args.lstm_classifier_learning_rate},
            ],
            weight_decay=args.weight_decay,
            betas=(0.9, 0.99),
        )

    best_score = float("-inf")
    best_path = output_dir / "best_model.pt"
    history: List[Dict[str, object]] = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, loaders["train"], optimizer, device)
        dev_metrics, _ = evaluate(model, loaders["dev"], device)
        epoch_result = {"epoch": epoch, "train_loss": train_loss, "dev": dev_metrics}
        history.append(epoch_result)
        print(json.dumps(epoch_result, ensure_ascii=False))

        score = float(dev_metrics[args.selection_metric])
        if score > best_score:
            best_score = score
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name_or_path": args.model_name_or_path,
                    "dropout": args.dropout,
                    "model_type": args.model_type,
                    "vocab": vocab,
                    "selection_metric": args.selection_metric,
                    "score": score,
                    "epoch": epoch,
                },
                best_path,
            )

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    dev_metrics, dev_rows = evaluate(model, loaders["dev"], device)
    test_metrics, test_rows = evaluate(model, loaders["test"], device)

    if tokenizer is not None:
        tokenizer.save_pretrained(output_dir / "tokenizer")
    save_json(output_dir / "history.json", {"epochs": history})
    save_json(output_dir / "dev_metrics.json", dev_metrics)
    save_json(output_dir / "test_metrics.json", test_metrics)
    save_predictions(output_dir / "dev_predictions.tsv", dev_rows)
    save_predictions(output_dir / "test_predictions.tsv", test_rows)
    save_json(output_dir / "run_config.json", vars(args))
    print(json.dumps({"best_epoch": checkpoint["epoch"], "test": test_metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
