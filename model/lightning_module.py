# imports
import lightning as pl
import torch

from .model_factory import create_model
from config import ModelConfig

# Define the LightningModule
class LightningModule(pl.LightningModule):
    def __init__(self, hparams, config: ModelConfig = None):
        """
        Initializes the LightningModule.

        Args:
            hparams (dict): Hyperparameters for the model.
        """
        super().__init__()
        
        model_hparams = vars(config) if config else {}
        hparams = {**hparams, **model_hparams}
        self.save_hyperparameters(hparams)
        self.model = create_model(**model_hparams)
    def forward(self, x):
        """
        Defines the forward pass of the model.
        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.model(x)

    def training_step(self, train_batch, batch_idx):
        """
        Defines the training step of the model.

        Args:
            train_batch (tuple): Batch of input and output tensors.
            batch_idx (int): Index of the batch.

        Returns:
            dict: Dictionary containing the loss.
        """
        x, y = train_batch
        output = self.forward(x)
        loss = torch.nn.CrossEntropyLoss()(output, y)
        self.log('train_loss', loss, sync_dist=True)
        return {'loss': loss}

    def validation_step(self, val_batch, batch_idx):
        """
        Defines the validation step of the model.

        Args:
            val_batch (tuple): Batch of input and output tensors.
            batch_idx (int): Index of the batch.

        Returns:
            None
        """
        x, y = val_batch
        output = self.forward(x)
        loss = torch.nn.CrossEntropyLoss()(output, y)
        _, predicted = torch.max(output, 1)
        accuracy = (predicted == y).sum().item() / len(x)
        self.log('val_loss', loss, sync_dist=True)
        self.log('val_acc', accuracy, sync_dist=True)

    def test_step(self, test_batch, batch_idx):
        """
        Defines the prediction step of the model.

        Args:
            test_batch (tuple): Batch of input tensors.
            batch_idx (int): Index of the batch.

        Returns:
            list: List of predicted class indices.
        """
        x = test_batch
        output = self.forward(x)
        _, predicted = torch.max(output, 1)
        return predicted

    def configure_optimizers(self):
        """
        Configures the optimizer for the model.

        Returns:
            torch.optim.Adam: Adam optimizer for the model.
        """
        optimizer = torch.optim.Adam(self.model.parameters(), self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size = self.trainer.estimated_stepping_batches * 2, 
            gamma=0.1
            )
        return [optimizer], [scheduler]

