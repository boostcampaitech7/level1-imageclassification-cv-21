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
        self.data_path = "/data/ephemeral/home/data/"
        # self.transform_mode = 'albumentation'


class ExperimentConfig:
    """Experiment-related configuration."""
    def __init__(self):
        self.save_dir = "/data/ephemeral/home/logs/"
        self.num_gpus = 1
        self.max_epochs = 100
        self.num_workers = 1  # number of workers in scheduling
        self.num_samples = 10  # number of workers in ray tune
        # self.checkpoint_interval = 5  # number of intervals to save checkpoint in pbt.
        self.ddp = False


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
        flattened_dict = {}
        for key, value in vars(self).items():
            if key != 'search_space' and key != 'training' and hasattr(value, '__dict__'):
                for subkey, subvalue in vars(value).items():
                    flattened_dict[f"{key}_{subkey}"] = subvalue
        return flattened_dict
        
