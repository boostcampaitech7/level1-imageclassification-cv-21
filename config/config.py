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
        self.transform_type = 'albumentations'
        self.num_workers = 3


class ExperimentConfig:
    """Experiment-related configuration."""
    def __init__(self):
        self.save_dir = "/data/ephemeral/home/logs/"
        self.num_gpus = 1
        self.num_workers = 6  # number of workers in scheduling
        self.num_samples = 10  # number of workers in ray tune
        self.ddp = False
        
        # Configs related with ASHA scheduler
        self.max_epochs = 100
        self.grace_period=10
        self.reduction_factor=2
        self.brackets=3

        # Manual checkpoint load and test option
        self.checkpoint_path = None


class Config:
    """Main configuration class."""
    def __init__(self):
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.dataset = DatasetConfig()
        self.experiment = ExperimentConfig()
        
        self.search_space = vars(self.training)
    
    
    def update_from_args(self, args):
        def update_config(obj, args):
            for key, value in vars(obj).items():
                if hasattr(value, '__dict__'):
                    update_config(value, args)
                else:
                    for arg_key, arg_value in vars(args).items():
                        if key == arg_key.replace("-", "_") and arg_value is not None:
                            setattr(obj, key, arg_value)

        update_config(self, args)
        
