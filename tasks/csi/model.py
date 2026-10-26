import torch
from torch import nn
from transformers import AutoModel


class BertSentenceClassifier(nn.Module):
    def __init__(self, model_name_or_path: str, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name_or_path)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, 2)

    def forward(self, **encoded_inputs):
        outputs = self.encoder(**encoded_inputs)
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is None:
            pooled = outputs.last_hidden_state[:, 0]
        return self.classifier(self.dropout(pooled))
