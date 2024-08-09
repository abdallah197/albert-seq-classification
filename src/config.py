from dataclasses import dataclass


@dataclass
class TrainConfig:
    file_path = '../data/imdb.csv'
    model_id = 'albert/albert-base-v2'
    batch_size = 4
    eval_steps = 10
    max_lr = 3e-5
    epochs = 1
    device = 'cpu'
    out_dir = 'output'
    train = False
