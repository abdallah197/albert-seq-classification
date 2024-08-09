import pandas as pd
import torch
from torch.utils.data import Dataset, random_split
from transformers import AutoTokenizer

### config ###
file_path = '../data/imdb.csv'
model_id = 'albert/albert-base-v2'

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
train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_dataloader, test_dataloader = prepare_dataloaders(dataset, split_size= 0.9)