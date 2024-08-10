import os
import time

from tqdm import tqdm
import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

from config import TrainConfig
from model import AlbertModelForClassification


def train(config: TrainConfig, model: AlbertModelForClassification, train_dataloader: DataLoader,
          test_dataloader: DataLoader):
    if not os.path.exists(config.out_dir):
        os.makedirs(config.out_dir)

    model = model.to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.max_lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, config.max_lr, epochs=config.epochs,
                                                    steps_per_epoch=len(train_dataloader))

    best_accuracy = 0.0
    best_model = model
    if not config.train:
        return best_model

    start_time = time.time()
    for epoch in range(config.epochs):
        print(f'Epoch: {epoch + 1}/{config.epochs}')
        total_loss = 0.0
        model.train()

        progress_bar = tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}")
        for step, item in enumerate(progress_bar):
            item = {key: val.to(config.device) for key, val in item.items()}
            optimizer.zero_grad()
            logits, loss = model(item)
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            if (step + 1) % config.eval_steps == 0:
                avg_loss = total_loss / config.eval_steps
                progress_bar.set_postfix({"Train Loss": f"{avg_loss:.4f}"})
                total_loss = 0.0
                eval_accuracy = evaluate(model, test_dataloader, config.device)
                print(f'Step: {step + 1}, Evaluation Accuracy: {eval_accuracy * 100:.4f}%')

                if eval_accuracy > best_accuracy:
                    best_accuracy = eval_accuracy
                    torch.save(model.state_dict(), f"{config.out_dir}/best_model_epoch_{epoch + 1}_step_{step + 1}.pth")
                    best_model = model.state_dict()

                model.train()

    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Best accuracy: {best_accuracy * 100:.4f}%")

    return best_model


def evaluate(model, dataloader, device):
    model.eval()
    preds, true_labels = [], []
    with torch.no_grad():
        for item in dataloader:
            item = {key: val.to(device) for key, val in item.items()}
            labels = item['labels'].cpu().tolist()
            scores, _ = model(item)
            preds_ = torch.argmax(scores, dim=1).cpu().tolist()
            preds.extend(preds_)
            true_labels.extend(labels)
    return accuracy_score(true_labels, preds)
