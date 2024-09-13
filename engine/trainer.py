# imports
import pytorch_lightning as pl
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import torch
import os

# Define the LightningModule
class MyLightningModule(pl.LightningModule):
    def __init__(self, hparams):
        """
        Initializes the LightningModule.

        Args:
            hparams (dict): Hyperparameters for the model.
        """
        super().__init__()
        self.hparams = hparams
        model_path = os.path.join('models', hparams['model_name'] + '.pth')
        self.model = torch.load(model_path)

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
        self.log('train_loss', loss)
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
        self.log('val_loss', loss)
        self.log('val_acc', accuracy)

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

    def test_epoch_end(self, outputs):
        """
        Saves the predicted labels to a CSV file.
        """
        predictions = torch.cat(outputs)
        predictions = predictions.cpu().numpy()
        
        # Assuming test_info is a pandas DataFrame with image paths
        test_info = pd.read_csv('/home/data/test.csv')
        
        # Create a new column for the predictions
        test_info['target'] = predictions
        
        # Reset index and rename the 'index' column to 'ID'
        test_info = test_info.reset_index().rename(columns={"index": "ID"})
        
        # Save to CSV
        file_name = f"{self.hparams['model_name']}_{date.today()}.csv"
        test_info.to_csv(file_name, index=False, lineterminator='\n')




    def configure_optimizers(self):
        """
        Configures the optimizer for the model.

        Returns:
            torch.optim.Adam: Adam optimizer for the model.
        """
        return torch.optim.Adam(self.model.parameters(), self.hparams['lr'])

