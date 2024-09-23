from .config import Config


class EfficientNet(Config):
    def __init__(self):
        super().__init__()
        self.model_name = "EfficientNet"  # Override the baseline model name
