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
            grace_period=self.config.experiment.grace_period, 
            reduction_factor=self.config.experiment.reduction_factor,
            brackets=self.config.experiment.brackets,
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
            num_workers=1,
            use_gpu=True,
            trainer_resources={"CPU": 0},
            resources_per_worker={
                "CPU": 6/self.config.experiment.num_workers, 
                "GPU": 1/self.config.experiment.num_workers
                },
        )
        return scaling_config
    def _define_run_config(self):
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
        run_config = RunConfig(
            name=f"{self.config.model.model_name}_tune_runs_{current_time}",
            checkpoint_config=CheckpointConfig(
                num_to_keep=2,
                checkpoint_score_attribute="val_loss",
                checkpoint_score_order="min",
            ),
            storage_path=f"{self.config.experiment.save_dir}/ray_results",
            callbacks=[WandbLoggerCallback(project=self.config.model.model_name)],
            verbose=1,
        )
        return run_config
    def _define_pltrainer(self):
        if self.config.experiment.ddp:
            trainer = Trainer(
                max_epochs=self.config.experiment.max_epochs,
                devices='auto',
                accelerator='auto',
                strategy=RayDDPStrategy(),
                callbacks=[RayTrainReportCallback()],
                plugins=[RayLightningEnvironment()],
                enable_progress_bar=False,
                )
            
            trainer = prepare_trainer(trainer)
        else:
            trainer = Trainer(
                max_epochs=self.config.experiment.max_epochs,
                devices=self.config.experiment.num_gpus,
                accelerator='auto',
                strategy='auto',
                callbacks=[RayTrainReportCallback()],
                enable_checkpointing=False,
                enable_progress_bar=False,
                )

        return trainer

    def _train_func(self, hparams): 
        # Create the dataloaders
        train_loader, val_loader = get_dataloaders(
            data_path=self.config.dataset.data_path, 
            transform_type=self.config.dataset.transform_type,
            batch_size=hparams['batch_size'],
            num_workers=self.config.dataset.num_workers
            )
        model = LightningModule(hparams, config=self.config.model)

        trainer = self._define_pltrainer()
        
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    def tune_and_train(self):
        tuner = tune.Tuner(
            self.ray_trainer, 
            param_space={"train_loop_config": self.config.search_space},  # Hyperparameter search space
            tune_config=self._define_tune_config(),  # Tuner configuration
            )
        result_grid = tuner.fit() ## Actual training happens here
        return result_grid
