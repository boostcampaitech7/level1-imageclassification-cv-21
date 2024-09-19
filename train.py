# imports
import argparse

import torch

from config.config_factory import get_config
from engine.tuner import RayTuner
from engine.test_runner import run_test


def tune_train_and_test(config):
    with RayTuner(config) as ray_tuner:
        result_grid = ray_tuner.tune_and_train() 
        # Get the best trial
        best_result = result_grid.get_best_result(metric="val_loss", mode="min")

    # Conduct testing with the best model loaded
    with best_result.checkpoint.as_directory() as ckpt_dir:
        run_test(config, ckpt_dir)

def main(config):
    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA not available. This program requires a CUDA-enabled NVIDIA GPU.")

    tune_train_and_test(config)


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Model training and hyperparameter tuning.')
    parser.add_argument('--model-name', type=str, help='Name of the model to use.')
    parser.add_argument('--num-gpus', type=int, help='Name of the model to use.')
    parser.add_argument('--smoke-test', action='store_true', help='Perform a small trial to test the setup.')
    parser.add_argument('--ddp', action='store_true', help='Perform the distributed data parallel. Only use when you have multiple gpus.')
    args = parser.parse_args()

    # Initialize and configure the model configuration object
    ConfigClass = get_config(args.model_name)
    config = ConfigClass()
    for key, value in vars(args).items():
        if hasattr(config, key) and value is not None:
            setattr(config, key)

    # Perform smoke test if enabled
    if args.smoke_test:
        config.experiment.max_epochs = 1
        config.experiment.num_samples = 1
        config.training.num_gpus = 1



    main(config)
