# imports
import os
import argparse
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining
from engine.trainer import MyLightningModule
import torch
from pytorch_lightning import Trainer
from config_factory import get_config
from dataset.dataloader import get_dataloaders, get_test_loader  # Import the dataloader function
import ray
from ray.air.integrations.wandb import WandbLoggerCallback


# Define the objective function to optimize
def train(config):
    # Create the dataloaders
    train_loader, val_loader = get_dataloaders(batch_size=config['batch_size'],
                                                              num_workers=1)
    model = MyLightningModule(config)

    trainer = Trainer(
        max_epochs=config['max_epochs'],
        accelerator='ddp',
        devices=config['num_gpus'],
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Access the logged values using the trainer object
    logs = trainer.callback_metrics
    val_loss = logs['val_loss'].item()
    val_acc = logs['val_acc'].item()

    tune.report(val_loss=val_loss, val_acc=val_acc)



# Define the tune run function
def tune_run(config):
    # Define the population-based training scheduler
    pbt = PopulationBasedTraining(
        time_attr="training_iteration", 
        perturbation_interval=5,
        metric="mean_accuracy",
        mode="max",
        hyperparam_mutations=config.search_space,
    )


    ray.init(local_mode=False)
    tuner = tune.Tuner(
        tune.with_resources(
            train, 
            resources={"cpu": 4, "gpu": 1},
        ),
        param_space={"model_name":config.model_name, "save_dir": config.save_dir, "max_epochs": config.max_epochs,}|config.search_space,
        tune_config=tune.TuneConfig(
        scheduler=pbt, 
        num_samples=config.num_samples,
        ),
        run_config=ray.train.RunConfig(
            name=f"{config.model_name}_tune_runs",
            checkpoint_config=ray.train.CheckpointConfig(checkpoint_score_attribute="mean_accuracy", num_to_keep=4,),
            storage_path="/tmp/ray_results",
            callbacks=[WandbLoggerCallback(project=config.model_name)],
            verbose=1,
        ),
    )

    tuner.fit()

    # Get the best trial
    best_trial = tuner.get_best_result(metric="val_loss", mode="min")

    # Get the checkpoint of the best trial
    best_checkpoint_path = best_trial.config.checkpoint

    # Load the best model checkpoint
    best_model = MyLightningModule.load_from_checkpoint(best_checkpoint_path)

    # Call the test loader
    test_loader = get_test_loader(data_path=config['data_path'], batch_size=config['batch_size'], num_workers=1)

    # Define the trainer for testing
    trainer_test = Trainer(gpus=1 if torch.cuda.is_available() else 0)
    trainer_test.test(best_model, test_dataloaders=test_loader)

    # Save the best model
    best_model_dir = os.path.join(config['save_dir'], "best_model")
    os.makedirs(best_model_dir, exist_ok=True)
    best_model_path = os.path.join(best_model_dir, "model.ckpt")
    torch.save(best_model.state_dict(), best_model_path)

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
    parser.add_argument('--model_name', type=str, help='Name of the model to use.')
    parser.add_argument('--num_gpus', type=int, help='Name of the model to use.')
    parser.add_argument('--smoke_test', action='store_true', help='Perform a small trial to test the setup.')
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
