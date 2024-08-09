import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

from config import TrainConfig
from model import AlbertModelForClassification


def train(config: TrainConfig, model: AlbertModelForClassification, train_dataloader: DataLoader,
          test_dataloader: DataLoader):
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.max_lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, config.max_lr, epochs=config.epochs,
                                                    steps_per_epoch=len(train_dataloader))

    best_accuracy = 0.0
    best_model = model
    if not config.train:
        return best_model

    for epoch in range(config.epochs):
        print(f'Epoch: {epoch}')
        total_loss = 0.0

        for step, item in enumerate(train_dataloader):
            model.train()
            item = {key: val.to(config.device) for key, val in item.items()}
            optimizer.zero_grad()
            logits, loss = model(item)
            total_loss += loss.detach()
            print(f'train loss: {total_loss.detach().item() / config.eval_steps}')
            loss.backward()
            optimizer.step()
            scheduler.step()

            if step % config.eval_steps:
                print(f'train loss: {total_loss.detach().item() / config.eval_steps}')
                model.eval()
                preds, true_labels = [], []

                for _, eval_item in enumerate(test_dataloader):
                    eval_item = {key: val.to(config.device) for key, val in eval_item.items()}
                    labels_ = eval_item['labels']
                    with torch.no_grad():
                        scores, _ = model(eval_item)
                    preds_ = torch.argmax(scores, dim=1).tolist()
                    preds.extend(preds_)
                    true_labels.extend(labels_)
                acc = accuracy_score(preds, true_labels)
                print(f'step: {step}, acc: {acc}')
                if acc > best_accuracy:
                    best_accuracy = acc
                    best_model = torch.save(model.state_dict(), config.out_dir)
    return best_model
