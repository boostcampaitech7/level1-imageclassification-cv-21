from engine.callbacks import PredictionCallback
from engine.lightning_module import LightningModule
from lightning import Trainer
from dataset.dataloader import get_test_loader  

def run_test(config, model, ckpt_dir):
    # Call the test loader
    test_loader = get_test_loader(data_path=config.data_path, batch_size=64, num_workers=6)
    
    # Define the trainer for testing
    pred_callback = PredictionCallback(f"{config.data_path}/test.csv", ckpt_dir, config.model_name)
    trainer_test = Trainer(callbacks=[pred_callback], logger=False)
    best_model = LightningModule.load_from_checkpoint(f"{ckpt_dir}/pltrainer.ckpt")
    # Conduct testing with the loaded model
    trainer_test.test(best_model, dataloaders=test_loader)