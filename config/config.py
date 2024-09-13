# config.py
import ray
from ray import tune

class Config:
    def __init__(self):
        self.model_name = "ResNet18"  # Baseline model
        self.save_dir = "/home/logs/"
        self.data_path = "/home/data/"
        self.batch_size = tune.choice([32, 64, 128])
        self.max_epochs = 1
        self.lr = tune.uniform(0.001, 0.1)
        self.weight_decay = tune.uniform(0.001, 0.1)
        self.n_estimators = tune.randint(10, 100)

        self.search_space = {
            'model_name': self.model_name,
            'batch_size': self.batch_size,
            'lr': self.lr,
            'weight_decay': self.weight_decay,
            'n_estimators': self.n_estimators,
        }
