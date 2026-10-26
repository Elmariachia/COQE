from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SentenceExample:
    sample_id: int
    text: str
    label: int


def read_coqe_split(path: str | Path) -> List[SentenceExample]:
    """Read sentence labels while skipping one or more tuple rows per sentence."""
    path = Path(path)
    examples: List[SentenceExample] = []
    for line_number, raw_line in enumerate(path.open(encoding="utf-8"), start=1):
        line = raw_line.rstrip("\r\n")
        if not line.strip() or "\t" not in line:
            continue

        text, raw_label = line.rsplit("\t", 1)
        raw_label = raw_label.strip()
        if raw_label not in {"0", "1"}:
            continue
        try:
            label = int(raw_label)
        except ValueError as exc:
            raise ValueError(f"Invalid label {raw_label!r} at {path}:{line_number}") from exc

        examples.append(SentenceExample(sample_id=len(examples), text=text, label=label))

    if not examples:
        raise ValueError(f"No sentence labels found in {path}")

    return examples


class SentenceDataset(Dataset):
    def __init__(self, examples: List[SentenceExample]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> SentenceExample:
        return self.examples[index]


class BertBatchCollator:
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, examples: List[SentenceExample]):
        texts = [example.text for example in examples]
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        labels = torch.tensor([example.label for example in examples], dtype=torch.long)
        return encoded, labels, examples


def build_char_vocab(examples: List[SentenceExample]) -> Dict[str, int]:
    vocab = {"<pad>": 0, "<unk>": 1}
    for example in examples:
        for char in example.text:
            if char not in vocab:
                vocab[char] = len(vocab)
    return vocab


class LstmBatchCollator:
    def __init__(self, vocab: Dict[str, int], max_length: int):
        self.vocab = vocab
        self.max_length = max_length

    def __call__(self, examples: List[SentenceExample]):
        sequences = []
        for example in examples:
            sequence = [self.vocab.get(char, 1) for char in example.text[: self.max_length]]
            sequences.append(sequence or [1])

        lengths = torch.tensor([len(sequence) for sequence in sequences], dtype=torch.long)
        input_ids = torch.zeros((len(sequences), int(lengths.max())), dtype=torch.long)
        for index, sequence in enumerate(sequences):
            input_ids[index, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)

        labels = torch.tensor([example.label for example in examples], dtype=torch.long)
        return {"input_ids": input_ids, "lengths": lengths}, labels, examples
