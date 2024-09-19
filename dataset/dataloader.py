# dataset/dataloader.py
import os

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, SubsetRandomSampler

from .dataset import CustomDataset
from .transforms import get_transforms


def get_dataloaders(data_path='/home/data/', batch_size=32, num_workers=1):
    """
    Returns train and validation data loaders.

    Args:
        data_path (str): Path to the dataset. Defaults to '/home/data/'.
        batch_size (int): Batch size for each data loader. Defaults to 32.
        num_workers (int): Number of worker threads for each data loader. Defaults to 1.

    Returns:
        Tuple[DataLoader, DataLoader]: Train and validation data loaders.
    """
    info_df = pd.read_csv(os.path.join(data_path, 'train.csv'))
    data_path = os.path.join(data_path, 'train')

    train_df, val_df = train_test_split(
        info_df, 
        test_size=0.2,
        stratify=info_df['target']
        )

    train_transform = get_transforms(mode='train')
    train_dataset = CustomDataset(data_path, train_df, transform=train_transform)

    # Create val dataset with validation transforms
    val_transform = get_transforms(mode='basic')
    val_dataset = CustomDataset(data_path, val_df, transform=val_transform)

    # Create samplers for the train and validation sets
    # train_sampler = SubsetRandomSampler(train_indices)
    # val_sampler = SubsetRandomSampler(val_indices)

    # Use DataCollator to create batches for both datasets
    train_loader = DataLoader(train_dataset,
                                batch_size=batch_size,
                                # sampler=train_sampler,
                                shuffle=True,
                                num_workers=num_workers,
                                pin_memory=True)
    val_loader = DataLoader(val_dataset,
                              batch_size=batch_size,
                            #   sampler=val_sampler,
                              shuffle=False,
                              num_workers=num_workers,
                             pin_memory=True)
    return train_loader, val_loader


def get_test_loader(data_path='/home/data/', batch_size=32, num_workers=1):
    """
    Returns a test data loader.

    Args:
        data_path (str): Path to the dataset. Defaults to '/home/data/'.
        batch_size (int): Batch size for the data loader. Defaults to 32.
        num_workers (int): Number of worker threads for the data loader. Defaults to 1.

    Returns:
        DataLoader: Test data loader.
    """
    test_df = pd.read_csv(os.path.join(data_path, 'test.csv'))
    data_path = os.path.join(data_path, 'test')
    transform = get_transforms(mode='test')
    test_dataset = CustomDataset(data_path, test_df, transform=transform, is_inference=True)
    test_loader = DataLoader(test_dataset,
                               batch_size=batch_size,
                               num_workers=num_workers,
                               pin_memory=True)
    return test_loader


# Example usage:
# train_loader, val_loader = get_dataloaders(data_path='/home/data/', batch_size=32, num_workers=4)
# test_loader = get_test_loader(data_path='/home/data/', batch_size=32, num_workers=4)

