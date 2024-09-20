from .config import Config
from .custom_nn_config import CustomNNConfig
from .coatnet_config import CoAtNetConfig
CONFIG_MAP = {
    'ResNet18': Config,
    'CustomNN': CustomNNConfig,
    'CoAtNet': CoAtNetConfig,
}

def get_config(model_name):
    """
    Retrieves a config class based on the provided model name.

    Args:
        model_name (str): The name of the model.

    Returns:
        Config or subclass: The config class associated with the model name.

    Raises:
        KeyError: If the model name is not found in the config map.
    """
    if model_name is None:
        model_name = "ResNet18"
    try:
        return CONFIG_MAP[model_name]
    except KeyError:
        raise ValueError(f"Unsupported model name: {model_name}")
