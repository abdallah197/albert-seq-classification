import torch
from transformers import AutoTokenizer


def inference(model, config, example_txt):
    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    tokenized_txt = tokenizer(example_txt, truncation=True, max_length=tokenizer.model_max_length,
                              padding='max_length', return_tensors='pt')
    tokenized_txt = {item: val.to(config.device) for item, val in tokenized_txt.items()}
    props, _ = model(tokenized_txt)
    preds = torch.argmax(props, dim=1).detach().item()
    return preds == 1
