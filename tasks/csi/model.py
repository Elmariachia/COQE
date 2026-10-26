import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class BertSentenceClassifier(nn.Module):
    def __init__(self, model_name_or_path: str, dropout: float = 0.1):
        super().__init__()
        from transformers import AutoModel

        self.encoder = AutoModel.from_pretrained(model_name_or_path)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, 2)

    def forward(self, **encoded_inputs):
        outputs = self.encoder(**encoded_inputs)
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is None:
            pooled = outputs.last_hidden_state[:, 0]
        return self.classifier(self.dropout(pooled))


class LstmSentenceClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 300,
        hidden_size: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.encoder = nn.LSTM(
            embedding_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, 2)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor):
        embedded = self.dropout(self.embedding(input_ids))
        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (hidden, _) = self.encoder(packed)
        pooled = torch.cat((hidden[-2], hidden[-1]), dim=-1)
        return self.classifier(self.dropout(pooled))
