import os
from datetime import datetime
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from lightning.pytorch.callbacks import Callback
from ray import train
from ray.train import Checkpoint

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