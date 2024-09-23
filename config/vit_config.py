from ray import tune

from .config import Config

class ViTConfig(Config):
    def __init__(self):
        super().__init__()
        self.model_name = "ViT"  # Override the baseline model name
        self.model.drop_path_rate = 0.2