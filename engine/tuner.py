# tuner.py
from datetime import datetime
import ray
from ray import train, tune
# from ray.tune.schedulers import PopulationBasedTraining
# from ray.tune.schedulers improt ASHAScheduler 
from ray.tune.schedulers.pb2 import PB2
from ray.air.integrations.wandb import WandbLoggerCallback
from lightning import Trainer
from ray.train import RunConfig, CheckpointConfig

from dataset import get_dataloaders
from model import LightningModule
from engine import CustomRayTrainReportCallback


def train_func(config_dict):  # Note that config_dict is dict here passed by pbt schduler
    # Create the dataloaders
    train_loader, val_loader = get_dataloaders(
        data_path=config_dict['dataset']['data_path'], 
        batch_size=config_dict['training']['batch_size'],
        num_workers=config_dict['experiment']['num_workers']
        )
    model = LightningModule(config_dict)

    trainer = Trainer(
        max_epochs=config_dict['experiment']['max_epochs'],
        accelerator='auto',
        devices=config_dict['experiment']['num_gpus'],
        strategy='ddp',
        logger=False,
        callbacks=[CustomRayTrainReportCallback(checkpoint_interval=config_dict['experiment']['checkpoint_interval'])],
        enable_progress_bar=False,
        enable_checkpointing=False,
        )
    
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


class RayTuner:
    def __init__(self, config):
        self.config = config  # config is Config class here consisting of 4 subclass config

    def __enter__(self):
        if ray.is_initialized():
            ray.shutdown()
        ray.init(local_mode=False)
        return self
    
    def __exit__(self, type, value, trace_back):
        ray.shutdown()

    def _define_scheduler(self):
        # Define the population-based training scheduler
        # pbt_scheduler = PopulationBasedTraining(
        #     time_attr="training_iteration",
        #     perturbation_interval=self.config.experiment.checkpoint_interval,
        #     metric="val_loss",
        #     mode="min",
        #     hyperparam_mutations=self.config.search_space,
        # )
        pbt_scheduler = PB2(
            time_attr="training_iteration",
            perturbation_interval=self.config.experiment.checkpoint_interval,
            metric="val_loss",
            mode="min",
            hyperparam_bounds=self.config.search_space,
        )
        return pbt_scheduler

    def _define_tune_config(self):
        tune_config = tune.TuneConfig(
            scheduler=self._define_scheduler(),
            num_samples=self.config.experiment.num_samples,
        )
        return tune_config

    def _define_run_config(self):
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
        run_config = RunConfig(
            name=f"{self.config.model.model_name}_tune_runs_{current_time}",
            checkpoint_config=CheckpointConfig(
                num_to_keep=10,
                checkpoint_score_attribute="val_loss",
                checkpoint_score_order="min",
            ),
            storage_path=f"{self.config.experiment.save_dir}/ray_results",
            callbacks=[WandbLoggerCallback(project=self.config.model.model_name)],
            verbose=1,
        )
        return run_config

    def tune_and_train(self):
        param_space = self.config.to_nested_dict2()
        tuner = tune.Tuner(
            tune.with_resources(
                train_func, 
                resources={
                    "cpu": 6/self.config.experiment.num_samples, 
                    "gpu": 1/self.config.experiment.num_samples
                    }
                ), 
            param_space=param_space,  # Hyperparameter search space
            tune_config=self._define_tune_config(),  # Tuner configuration
            run_config=self._define_run_config(),  # Run environment configuration
        )
        result_grid = tuner.fit() ## Actual training happens here
        return result_grid
