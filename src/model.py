import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class AlbertModelForClassification(nn.Module):
    def __init__(self, model_id, num_labels=2):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_id)
        # check documentation in hf
        hidden_size = self.model.config.hidden_size
        self.out = nn.Linear(hidden_size, num_labels)

    def forward(self, inputs, use_cls_token=True):
        labels = None
        loss = None
        if 'labels' in inputs:
            labels = inputs['labels']
            del inputs['labels']
        outputs = self.model(**inputs)

        # choosing cls of avg output, will change dependign on performance
        if use_cls_token:
            logits = outputs.pooler_output
        else:
            last_hidden_state = outputs.last_hidden_state
            logits = torch.mean(last_hidden_state, dim=1)
        scores = self.out(logits)
        scores = F.softmax(scores, dim=-1)
        if labels:
            loss = F.cross_entropy(scores, labels)
        return scores, loss
