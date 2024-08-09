import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer


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


def prepare_dataloaders(dataset, batch_size, split_size=0.9):
    train_size = int(split_size * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_dataloader = DataLoader(train_dataset, batch_size)
    test_dataloadder = DataLoader(test_dataset, batch_size)
    return train_dataloader, test_dataloadder
