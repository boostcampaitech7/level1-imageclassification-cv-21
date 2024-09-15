from lightning.pytorch.callbacks import Callback
import numpy as np
import pandas as pd
from datetime import date

class PredictionCallback(Callback):
    def __init__(self, data_path, model_name):
        self.predictions = []
        self.data_path = data_path
        self.model_name = model_name

    def on_test_batch_end(self, trainer, pl_module, outputs, *args, **kwargs):
        self.predictions.extend(outputs['test_step'].cpu().numpy())

    def on_test_end(self, *args, **kwargs):
        predictions = np.array(self.predictions)
        test_info = pd.read_csv(self.data_path)
        test_info['target'] = predictions
        test_info = test_info.reset_index().rename(columns={"index": "ID"})
        file_name = f"{self.model_name}_{date.today()}.csv"
        test_info.to_csv(file_name, index=False, lineterminator='\n')
        print("Output csv file successfully saved!!")