from ray import tune

class ModelConfig:
    """Model-related configuration."""
    def __init__(self):
        self.model_name = "ResNet18"  # Baseline model
        # self.num_layers = ~
        # self.num_heads = ~


class TrainingConfig:
    """Training-related configuration."""
    def __init__(self):
        self.batch_size = tune.choice([32, 64, 128])
        self.lr = tune.loguniform(0.001, 0.1)
        self.weight_decay = tune.loguniform(0.001, 0.1)
        


class DatasetConfig:
    """Dataset-related configuration."""
    def __init__(self):
        self.data_path = "/home/data/"
        # self.transform_mode = 'albumentation'


class ExperimentConfig:
    """Experiment-related configuration."""
    def __init__(self):
        self.save_dir = "/home/logs/"
        self.num_gpus = 1
        self.max_epochs = 100
        self.num_workers = 2  # number of cpus workers in dataloader
        self.num_samples = 4  # number of workers in population-based training(pbt)
        self.checkpoint_interval = 5  # number of intervals to save checkpoint in pbt.


class Config:
    """Main configuration class."""
    def __init__(self):
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.dataset = DatasetConfig()
        self.experiment = ExperimentConfig()

        self.search_space = {
            'batch_size': self.training.batch_size,
            'lr': self.training.lr,
            'weight_decay': self.training.weight_decay,
        }
    
    def flatten_to_dict(self):
        return {
            **vars(self.model),
            **vars(self.training),
            **self.search_space,
            **vars(self.dataset),
            **vars(self.experiment)
        }
    
    def to_nested_dict(self):
        return {
            'model': vars(self.model),
            'training': vars(self.training),
            'dataset': vars(self.dataset),
            'experiment': vars(self.experiment),
            **self.search_space
        }
        
