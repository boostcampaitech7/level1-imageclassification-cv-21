# tuner.py
from lightning import Trainer
import ray
from ray import train, tune
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from ray.air.integrations.wandb import WandbLoggerCallback
from dataset.dataloader import get_dataloaders
from engine.trainer import LightningModule


def train_func(config):
    # Create the dataloaders
    train_loader, val_loader = get_dataloaders(batch_size=config['batch_size'],
                                                              num_workers=config['num_workers'])
    model = LightningModule(config)

    trainer = Trainer(
        max_epochs=config['max_epochs'],
        accelerator='gpu',
        devices=config['num_gpus'],
        strategy='ddp',
        callbacks=[TuneReportCheckpointCallback(
            metrics={"val_loss": "val_loss", "val_acc": "val_acc"},
            filename="pltrainer.ckpt", on="validation_end",
            )],
        )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


class RayTuner:
    def __init__(self, config):
        self.config = config  # config is Config class here

    def __enter__(self):
        if ray.is_initialized():
            ray.shutdown()
        ray.init(local_mode=False)
    
    def __exit__(self):
        ray.shutdown()

    def _define_scheduler(self):
        # Define the population-based training scheduler
        pbt_scheduler = PopulationBasedTraining(
            time_attr="training_iteration", 
            perturbation_interval=self.config.checkpoint_interval,
            metric="val_loss",
            mode="min",
            hyperparam_mutations=self.config.search_space,
        )
        return pbt_scheduler

    def _define_tune_config(self):
        tune_config = tune.TuneConfig(
            scheduler=self._define_scheduler(), 
            num_samples=self.config.num_samples,
        )
        return tune_config

    def _define_run_config(self):
        run_config = train.RunConfig(
            name=f"{self.config.model_name}_tune_runs",
            checkpoint_config=train.CheckpointConfig(
                num_to_keep=4,
                checkpoint_score_attribute="val_loss", 
            ),
            storage_path="/tmp/ray_results",
            callbacks=[WandbLoggerCallback(project=self.config.model_name)],
            verbose=1,
        )
        return run_config

    def tune(self):
        self._init_ray()
        param_space = {**{key: value for key, value in vars(self.config).items() if key != 'search_space'}, 
                        **self.config.search_space}
        tuner = tune.Tuner(
            tune.with_resources(train_func, resources={"cpu": 4, "gpu": 0.5}), # TODO: What does with_resources do?
            param_space=param_space,  # Hyperparameter search space
            tune_config=self._define_tune_config(),  # Tuner configuration
            run_config=self._define_run_config(),  # Run environment configuration
        )
        result_grid = tuner.fit()
        return result_grid