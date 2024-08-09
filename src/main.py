import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, AutoModel

### config ###
file_path = '../data/imdb.csv'
model_id = 'albert/albert-base-v2'
batch_size = 16


class TextClassification(Dataset):
    def __init__(self, file_path, model_id):
        dataframe = pd.read_csv(file_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.text = dataframe.review.tolist()
        labels = dataframe.sentiment.tolist()
        label2id = {'positive': 1, 'negative': 0}
        self.labels = [label2id[label] for label in labels]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sentence = self.text[idx]
        label = self.labels[idx]

        tokenized_txt = self.tokenizer(sentence, truncation=True, max_length=self.tokenizer.model_max_length,
                                       padding='max_length', return_tensors='pt')

        # force a python dict type
        tokenized_txt = {item: val.squeeze() for item, val in tokenized_txt.items()}
        label = torch.tensor(label)
        tokenized_txt['labels'] = label.type_as(tokenized_txt['input_ids'])
        return tokenized_txt


dataset = TextClassification(file_path, model_id)


def prepare_dataloaders(dataset, batch_size, split_size=0.9):
    train_size = int(split_size * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_dataloader = DataLoader(train_dataset, batch_size)
    test_dataloadder = DataLoader(test_dataset, batch_size)
    return train_dataloader, test_dataloadder


train_dataloader, test_dataloader = prepare_dataloaders(dataset, batch_size)


class AlbertModelForClassification(nn.Module):
    def __init__(self, model_id, num_labels=2):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_id)
        # check documentation in hf
        hidden_size = self.model.config.hidden_size
        self.out = nn.Linear(hidden_size, num_labels)

    def forward(self, inputs, use_cls_token=True):
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
        loss = F.cross_entropy(scores, labels)
        return scores, loss


model = AlbertModelForClassification(model_id)

for item in train_dataloader:
    output, loss = model(item)
    print(output)
