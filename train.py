# imports
import os
from utils.logger import get_logger
# from ray import tune
from ray.tune.schedulers import PopulationBasedTraining
from engine.trainer import MyLightningModule
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from config.config import Config
from ray.tune import CLIReporter
from dataset.dataloader import get_dataloaders, get_test_loader  # Import the dataloader function
import ray

# Load the config
config = Config()

# Define the objective function to optimize
def train(config):
    # Create a logger
    logger = get_logger(config['save_dir'], type(MyLightningModule).__name__)

    # Create the dataloaders
    train_loader, val_loader = get_dataloaders(batch_size=config['batch_size'],
                                                              num_workers=1)

    model = MyLightningModule(config)
    early_stopper = EarlyStopping(
        monitor='val_loss',
        patience=5,
        min_delta=0.001,
        mode='min',
        check_finiteness=True,
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
        num_workers=1,
        accelerator='dp' if torch.cuda.is_available() else 'cpu',
        devices=torch.cuda.device_count() if torch.cuda.is_available() else 1,
        logger=logger,
        callbacks=[early_stopper, model_checkpoint],
        log_every_n_steps=10,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    # Report the metrics
    val_loss = model.val_loss
    val_acc = model.val_acc
    tune.report(val_loss=val_loss, val_acc=val_acc)

# Define the population-based training scheduler
pbt = PopulationBasedTraining(
    perturbation_interval=10,
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
        param_space=config.search_space,
        tune_config=tune.TuneConfig(
        metric="val_loss",
        mode="min",
        num_samples=3,
        sync_config=ray.train.SyncConfig(),
            trial_scheduler=pbt,
            checkpoint_freq=0,  # disable checkpointing for trials
        ),
        run_config=ray.train.RunConfig(
            storage_path=f"./{type(MyLightningModule).__name__}_tune_runs",
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
    test_loader = get_test_loader(data_path=config['data_path'], batch_size=config['batch_size'], num_workers=config['n_workers'])

    # Define the trainer for testing
    trainer_test = Trainer(gpus=1 if torch.cuda.is_available() else 0)
    test_outputs = trainer_test.test(best_model, test_dataloaders=test_loader)

    # Save the best model
    best_model_dir = os.path.join(config['save_dir'], "best_model")
    os.makedirs(best_model_dir, exist_ok=True)
    best_model_path = os.path.join(best_model_dir, "model.ckpt")
    torch.save(best_model.state_dict(), best_model_path)

    ray.shutdown()

# Define the main function
def main():
    tune_run()

if __name__ == "__main__":
    main()