from ray import tune
# from utils.logger import WandbLogger

class Config:
    def __init__(self):
        self.model_name = "ResNet18"  # Baseline model
        self.save_dir = "/home/logs/"
        self.data_path = "/home/data/"
        self.batch_size = tune.choice([32, 64, 128])
        self.max_epochs = 100
        self.num_samples = 4  # number of workers in population-based training
        self.num_workers = 2  # number of cpus workers in dataloader
        self.lr = tune.loguniform(0.001, 0.1)
        self.weight_decay = tune.loguniform(0.001, 0.1)

        self.search_space = {
            'batch_size': self.batch_size,
            'lr': self.lr,
            'weight_decay': self.weight_decay,
        }
    # def get_logger(self):
    #     return WandbLogger(project_name=self.model_name, config={
    #         "save_dir": self.save_dir,
    #         "batch_size": self.batch_size,
    #         "max_epochs": self.max_epochs,
    #         # add other config parameters as needed
    #     })