# imports
import os
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining
from engine.trainer import MyLightningModule
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from config.config import Config
from ray.tune import CLIReporter
from dataset.dataloader import get_dataloaders, get_test_loader  # Import the dataloader function
import ray
from ray.air.integrations.wandb import WandbLoggerCallback

# Load the config
config = Config()
# logger = config.get_logger()

# Define the objective function to optimize
def train(config):
    # Create the dataloaders
    train_loader, val_loader = get_dataloaders(batch_size=config['batch_size'],
                                                              num_workers=1)
    print(config, "This is config")
    model = MyLightningModule(config)
    early_stopper = EarlyStopping(
        monitor='val_loss',
        patience=5,
        min_delta=0.001,
        check_on_train_epoch_end=False
    )

    model_checkpoint = ModelCheckpoint(
        dirpath=f"{config['save_dir']}/{type(model).__name__}_checkpoints",
        filename='{epoch}-{val_loss:.2f}-{val_acc:.2f}',
        save_top_k=1,
        save_weights_only=False,
        mode='min',
        monitor='val_loss'
    )

    trainer = Trainer(
        max_epochs=config['max_epochs'],
        accelerator='dp' if torch.cuda.is_available() else 'cpu',
        devices=torch.cuda.device_count() if torch.cuda.is_available() else 1,
        # logger=logger,
        callbacks=[early_stopper, model_checkpoint],
        log_every_n_steps=10,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    # Report the metrics
    val_loss = model.val_loss
    val_acc = model.val_acc
    # logger.log_metrics({"val_loss": val_loss, "val_acc": val_acc})
    tune.report(val_loss=val_loss, val_acc=val_acc)

# Define the population-based training scheduler
pbt = PopulationBasedTraining(
    time_attr="training_iteration", 
    perturbation_interval=5,
    metric="mean_accuracy",
    mode="max",
    hyperparam_mutations=config.search_space,
)

# Define the reportable hyperparameters
reporter = CLIReporter(
    parameter_columns=["batch_size", "lr", "weight_decay", "n_estimators"],
    metric_columns=["val_loss", "val_acc"],
)

# Define the tune run function
def tune_run():
    ray.init(local_mode=False)
    tuner = tune.Tuner(
        tune.with_resources(
            train, 
            resources={"cpu": 4, "gpu": 1},
        ),
        param_space={"model_name":config.model_name, "save_dir": config.save_dir, "max_epochs": config.max_epochs,}|config.search_space,
        tune_config=tune.TuneConfig(
        # metric="val_loss",
        # mode="min",
        scheduler=pbt, 
        num_samples=1,
        ),
        run_config=ray.train.RunConfig(
            name=f"{type(MyLightningModule).__name__}_tune_runs",
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
    test_outputs = trainer_test.test(best_model, test_dataloaders=test_loader)

    # Save the best model
    best_model_dir = os.path.join(config['save_dir'], "best_model")
    os.makedirs(best_model_dir, exist_ok=True)
    best_model_path = os.path.join(best_model_dir, "model.ckpt")
    torch.save(best_model.state_dict(), best_model_path)

    # logger.finish()
    ray.shutdown()

# Define the main function
def main():
    tune_run()

if __name__ == "__main__":
    main()
