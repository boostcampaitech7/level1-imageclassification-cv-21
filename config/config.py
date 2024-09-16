from ray import tune

class Config:
    def __init__(self):
        self.model_name = "ResNet18"  # Baseline model
        self.save_dir = "/home/logs/"
        self.data_path = "/home/data/"
        self.batch_size = tune.choice([32, 64, 128])
        self.num_gpus = 1
        self.max_epochs = 100
        self.num_samples = 4  # number of workers in population-based training
        self.num_workers = 2  # number of cpus workers in dataloader
        self.checkpoint_interval = 5  # number of epoch 
        self.lr = tune.loguniform(0.001, 0.1)
        self.weight_decay = tune.loguniform(0.001, 0.1)

        self.search_space = {
            'batch_size': self.batch_size,
            'lr': self.lr,
            'weight_decay': self.weight_decay,
        }
