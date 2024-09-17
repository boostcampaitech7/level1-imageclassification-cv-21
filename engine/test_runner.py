from lightning import Trainer

from .callbacks import PredictionCallback
from dataset import get_test_loader 
from model import LightningModule 

def run_test(config, ckpt_dir):
    # Call the test loader
    test_loader = get_test_loader(data_path=config.dataset.data_path, batch_size=64, num_workers=6)
    
    # Define the trainer for testing
    pred_callback = PredictionCallback(f"{config.dataset.data_path}/test.csv", ckpt_dir, config.model.model_name)
    trainer_test = Trainer(callbacks=[pred_callback], logger=False, enable_progress_bar=False,)
    best_model = LightningModule.load_from_checkpoint(f"{ckpt_dir}/pltrainer.ckpt")
    # Conduct testing with the loaded model
    trainer_test.test(best_model, dataloaders=test_loader)
