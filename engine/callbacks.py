import os
from datetime import datetime
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from lightning.pytorch.callbacks import Callback
from ray import train
from ray.train import Checkpoint

class CustomRayTrainReportCallback(Callback):
    def __init__(self, 
                 checkpoint_interval=5):
        self.checkpoint_interval = checkpoint_interval

    def on_validation_end(self, trainer, pl_module):
        should_checkpoint = trainer.current_epoch % self.checkpoint_interval == 0
        with TemporaryDirectory() as tmpdir:
            # Fetch metrics
            metrics = trainer.callback_metrics
            metrics = {k: v.item() for k, v in metrics.items()}
            # Add customized metrics
            metrics["epoch"] = trainer.current_epoch
            
            checkpoint = None
            global_rank = train.get_context().get_world_rank() == 0
            if should_checkpoint and global_rank:
                # Save model checkpoint file to tmpdir
                ckpt_path = os.path.join(tmpdir, "ckpt.ckpt")
                trainer.save_checkpoint(ckpt_path, weights_only=False)
                checkpoint = Checkpoint.from_directory(tmpdir)

            # Report to train session
            train.report(metrics=metrics, checkpoint=checkpoint)



class PredictionCallback(Callback):
    def __init__(self, data_path, ckpt_dir, model_name):
        self.data_path = data_path
        self.ckpt_dir = ckpt_dir
        self.model_name = model_name
        self.predictions = []

    def on_test_batch_end(self, trainer, pl_module, outputs, *args, **kwargs):
        self.predictions.extend(outputs.cpu().numpy())

    def on_test_end(self, *args, **kwargs):
        predictions = np.array(self.predictions)
        test_info = pd.read_csv(self.data_path)
        test_info['target'] = predictions
        test_info = test_info.reset_index().rename(columns={"index": "ID"})
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
        file_name = os.path.join(self.ckpt_dir, f"{self.model_name}_predictions_{current_time}.csv")
        test_info.to_csv(file_name, index=False, lineterminator='\n')
        print(f"Output csv file successfully saved in {file_name}!!")