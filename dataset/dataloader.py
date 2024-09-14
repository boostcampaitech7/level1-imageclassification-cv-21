# dataset/dataloader.py
from torch.utils.data import DataLoader, SubsetRandomSampler
from dataset import CustomDataset
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

    transform = get_transforms(mode='train')
    full_dataset = CustomDataset(data_path, mode='train', transform=transform)

    # Create indices for the train and validation sets
    indices = list(range(len(full_dataset)))
    split = int(0.8 * len(indices))
    train_indices = indices[:split]
    val_indices = indices[split:]

    # Create val dataset with validation transforms
    val_dataset = CustomDataset(data_path, mode='train', transform=transform)

    # Create samplers for the train and validation sets
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    # Use DataCollator to create batches for both datasets
    train_loader = DataLoader(full_dataset,
                                batch_size=batch_size,
                                sampler=train_sampler,
                                num_workers=num_workers,
                                pin_memory=True)
    val_loader = DataLoader(val_dataset,
                              batch_size=batch_size,
                              sampler=val_sampler,
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
    transform = get_transforms(mode='test')
    test_dataset = CustomDataset(data_path, mode='test', transform=transform)
    test_loader = DataLoader(test_dataset,
                               batch_size=batch_size,
                               num_workers=num_workers,
                               pin_memory=True)
    return test_loader


# Example usage:
# train_loader, val_loader = get_dataloaders(data_path='/home/data/', batch_size=32, num_workers=4)
# test_loader = get_test_loader(data_path='/home/data/', batch_size=32, num_workers=4)

