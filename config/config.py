# config.py
import ray
from ray import tune
from utils.logger import WandbLogger

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
            'batch_size': self.batch_size,
            'lr': self.lr,
            'weight_decay': self.weight_decay,
            'n_estimators': self.n_estimators,
        }
    def get_logger(self):
        return WandbLogger(project_name=self.model_name, config={
            "save_dir": self.save_dir,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            # add other config parameters as needed
        })