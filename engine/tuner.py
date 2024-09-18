# tuner.py
from datetime import datetime
import ray
from ray import train, tune
from ray.tune.schedulers import ASHAScheduler 
from ray.air.integrations.wandb import WandbLoggerCallback
from lightning import Trainer
from ray.train import RunConfig, ScalingConfig, CheckpointConfig
from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)
from dataset import get_dataloaders
from model import LightningModule
from ray.train.torch import TorchTrainer

class RayTuner:
    def __init__(self, config):
        self.config = config  # config is Config class here consisting of 4 subclass config
        # Define a TorchTrainer without hyper-parameters for Tuner
        self.ray_trainer = TorchTrainer(
            self._train_func,
            train_loop_config=self.config.flatten_to_dict(),
            scaling_config=self._define_scaling_config(),
            run_config=self._define_run_config(),
        )
    def __enter__(self):
        if ray.is_initialized():
            ray.shutdown()
        ray.init(local_mode=False)
        return self
    
    def __exit__(self, type, value, trace_back):
        ray.shutdown()

    def _define_scheduler(self):
        scheduler = ASHAScheduler(
            max_t=self.config.experiment.max_epochs, 
            grace_period=1, 
            reduction_factor=3,
            brackets=3,
            )
        return scheduler

    def _define_tune_config(self):
        tune_config = tune.TuneConfig(
            metric="val_loss",
            mode="min",
            num_samples=self.config.experiment.num_samples,
            scheduler=self._define_scheduler(),
        )
        return tune_config
    
    def _define_scaling_config(self):
        scaling_config = ScalingConfig(
            num_workers=self.config.experiment.num_workers,
            use_gpu=True,
            resources_per_worker={"CPU": 5, "GPU": 1}
        )
        return scaling_config
    def _define_run_config(self):
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
        run_config = RunConfig(
            name=f"{self.config.model.model_name}_tune_runs_{current_time}",
            checkpoint_config=CheckpointConfig(
                num_to_keep=3,
                checkpoint_score_attribute="val_loss",
                checkpoint_score_order="min",
            ),
            storage_path=f"{self.config.experiment.save_dir}/ray_results",
            callbacks=[WandbLoggerCallback(project=self.config.model.model_name)],
            verbose=1,
        )
        return run_config
    @staticmethod
    def _train_func(config_dict):
        def flatten_to_nested(flattened_dict):
            # transforms the dict of the form {key}_{subkey}:value to nested dict.
            nested_dict = {'dataset': {}, 'model': {}, 'experiment': {}}
            expected_keys = ['dataset', 'model', 'experiment']
            for key, value in flattened_dict.items():
                if "_" in key:
                    parts = key.split("_")
                    subkey = '_'.join(parts[1:])
                    if parts[0] in expected_keys:
                        nested_dict[parts[0]][subkey] = value
                    else:
                        nested_dict[key] = value
                else:
                    nested_dict[key] = value
            return nested_dict
        
        config_dict = flatten_to_nested(config_dict)
        # Create the dataloaders
        train_loader, val_loader = get_dataloaders(
            data_path=config_dict['dataset']['data_path'], 
            batch_size=config_dict['batch_size'],
            num_workers=2
            )
        print(config_dict)
        model = LightningModule(config_dict)

        trainer = Trainer(
            devices='auto',
            accelerator='auto',
            strategy=RayDDPStrategy(),
            callbacks=[RayTrainReportCallback()],
            plugins=[RayLightningEnvironment()],
            enable_progress_bar=False,
            )
        
        trainer = prepare_trainer(trainer)
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    def tune_and_train(self):
        tuner = tune.Tuner(
            self.ray_trainer, 
            param_space={"train_loop_config": self.config.search_space},  # Hyperparameter search space
            tune_config=self._define_tune_config(),  # Tuner configuration
            )
        result_grid = tuner.fit() ## Actual training happens here
        return result_grid
