# imports
import os
import argparse
import torch
from lightning import Trainer
import ray
from ray import train, tune
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from ray.air.integrations.wandb import WandbLoggerCallback
from config.config_factory import get_config
from dataset.dataloader import get_dataloaders, get_test_loader  # Import the dataloader function
from engine.trainer import MyLightningModule



# Define the objective function to optimize
def train_func(config):
    # Create the dataloaders
    train_loader, val_loader = get_dataloaders(batch_size=config['batch_size'],
                                                              num_workers=config['num_workers'])
    model = MyLightningModule(config)

    checkpoint_callback = TuneReportCheckpointCallback(
        metrics={"val_loss": "val_loss", "val_acc": "val_acc"}, 
        filename="pltrainer.ckpt", on="validation_end",
    )

    trainer = Trainer(
        max_epochs=config['max_epochs'],
        accelerator='gpu',
        devices=config['num_gpus'],
        strategy='ddp',
        callbacks=[checkpoint_callback]
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)



# Define the tune run function
def tune_run(config):
    if ray.is_initialized():
        ray.shutdown()
    ray.init(local_mode=False)

    # Define the population-based training scheduler
    pbt_scheduler = PopulationBasedTraining(
        time_attr="training_iteration", 
        perturbation_interval=config.checkpoint_interval,
        metric="val_loss",
        mode="min",
        hyperparam_mutations=config.search_space,
    )


    param_space = {**{key: value for key, value in vars(config).items() if key != 'search_space'}, 
                   **config.search_space}


    tune_config = tune.TuneConfig(
        scheduler=pbt_scheduler, 
        num_samples=config.num_samples,
        )   
    
    run_config = ray.train.RunConfig(
        name=f"{config.model_name}_tune_runs",
        checkpoint_config=ray.train.CheckpointConfig(
            num_to_keep=4,
            checkpoint_score_attribute="val_loss", 
            ),
        storage_path="/tmp/ray_results",
        callbacks=[WandbLoggerCallback(project=config.model_name)],
        verbose=1,
        )  
    

    tuner = tune.Tuner(
        tune.with_resources(train_func, resources={"cpu": 4, "gpu": 0.5},),
        tune_config=tune_config,
        run_config=run_config,
        param_space=param_space,
    )

    result_grid = tuner.fit()


    # Get the best trial
    best_result = result_grid.get_best_result(metric="val_loss", mode="min")

    # Load the best model checkpoint
    with best_result.checkpoint.as_directory() as ckpt_dir:
        best_model = MyLightningModule.load_from_checkpoint(os.path.join(ckpt_dir, "pltrainer.ckpt"))

    # Call the test loader
    test_loader = get_test_loader(data_path=config['data_path'], batch_size=64, num_workers=6)

    # Define the trainer for testing
    trainer_test = Trainer()
    trainer_test.test(best_model, dataloaders=test_loader)


    ray.shutdown()

# Define the main function
def main(config):
    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA not available. This program requires a CUDA-enabled NVIDIA GPU.")

    tune_run(config)

# Define the main entry point of the script
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
    for key, value in vars(args):
        if hasattr(config, key) and value is not None:
            setattr(config, key)

    # Perform smoke test if enabled        
    if args.smoke_test:
        config.max_epochs = 1
        config.num_samples = 1
        config.num_gpus = 1
    
    main(config)
