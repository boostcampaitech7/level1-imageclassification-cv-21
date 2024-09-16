# imports
import argparse
import torch
from lightning import Trainer
from config.config_factory import get_config
from dataset.dataloader import get_dataloaders, get_test_loader  
from engine.trainer import LightningModule
from engine.callbacks import PredictionCallback
from engine.tuner import RayTuner


def test_model(config, ckpt_dir):
    # Call the test loader
    test_loader = get_test_loader(data_path=config.data_path, batch_size=64, num_workers=6)
    # Define the trainer for testing
    pred_callback = PredictionCallback(f"{config.data_path}/test.csv", ckpt_dir, config.model_name)
    trainer_test = Trainer(callbacks=[pred_callback], logger=False)
    return test_loader, trainer_test


def tune_and_test(config):
    with RayTuner(config) as ray_tuner:
        result_grid = ray_tuner.tune() 
        # Get the best trial
        best_result = result_grid.get_best_result(metric="val_loss", mode="min")

    # Load the best model checkpoint
    with best_result.checkpoint.as_directory() as ckpt_dir:
        best_model = LightningModule.load_from_checkpoint(f"{ckpt_dir}/pltrainer.ckpt")
        # Conduct testing with the best model loaded
        test_loader, trainer_test = test_model(config, ckpt_dir)
        trainer_test.test(best_model, dataloaders=test_loader)

def main(config):
    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA not available. This program requires a CUDA-enabled NVIDIA GPU.")

    tune_and_test(config)


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Model training and hyperparameter tuning.')
    parser.add_argument('--model-name', type=str, help='Name of the model to use.')
    parser.add_argument('--num-gpus', type=int, help='Name of the model to use.')
    parser.add_argument('--smoke-test', action='store_true', help='Perform a small trial to test the setup.')
    args = parser.parse_args()

    # Initialize and configure the model configuration object
    ConfigClass = get_config(args.model_name)
    config = ConfigClass()
    for key, value in vars(args).items():
        if hasattr(config, key) and value is not None:
            setattr(config, key)

    # Perform smoke test if enabled
    if args.smoke_test:
        config.max_epochs = 1
        config.num_samples = 1
        config.num_gpus = 1

    main(config)
