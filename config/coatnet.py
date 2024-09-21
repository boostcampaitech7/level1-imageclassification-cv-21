from ray import tune

from .config import Config


class CoAtNetConfig(Config):
    def __init__(self):
        super().__init__()
        self.model_name = "CoAtNet"  # Override the baseline model name
