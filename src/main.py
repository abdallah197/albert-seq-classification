import torch

from config import TrainConfig
from data import TextClassification, prepare_dataloaders
from inference import inference
from model import AlbertModelForClassification
from train import train

config = TrainConfig()

if torch.cuda.is_available():
    config.device = "cuda"
print(f"using device: {config.device}")

dataset = TextClassification(config.file_path, config.model_id)

train_dataloader, test_dataloader = prepare_dataloaders(dataset, config.batch_size)
model = AlbertModelForClassification(config.model_id)

if config.train:
    print('Running Training')

best_model = train(config, model, train_dataloader, test_dataloader)
example_txt = "Besides being boring, the scenes were oppressive and dark. The movie tried to portray some kind of moral, but fell flat with its message. What were the redeeming qualities?? On top of that, I don't think it could make librarians look any more unglamorous than it did."

print(inference(model, config, example_txt))
