from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import os

class CustomDataset(Dataset):
    def __init__(self, data_path, mode='train'):
        """
        Initializes the CustomDataset class.

        Args:
        - data_path (str): The path to the dataset.
        - mode (str): The mode of the dataset, either 'train' or 'test' (default: 'train').
        """
        self.data_path = data_path
        self.mode = mode
        self.data = []
        
        if mode == 'train':
            self.data = pd.read_csv(os.path.join(data_path, 'train.csv'))
            self.images = self.data['image_path']
            self.labels = self.data['target'] 
        else:
            self.images = pd.read_csv(os.path.join(data_path, 'test.csv'))['image_path']  # Read image paths from test.csv
    def __getitem__(self, index):
        """
        Returns the image and label at the specified index.

        Args:
        - index (int): The index of the image and label.

        Returns:
        - image: The image at the specified index.
        - label: The label at the specified index (None if in 'test' mode).
        """
        image_path = None
        label = None
        
        if self.mode == 'train':
            image_path = self.images[index]
            label = self.labels[index]
        else:
            image_path = self.images[index]
        
        image = Image.open(image_path)
        
        if self.mode == 'train':
            return image, label
        else:
            return image

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
        - int: The length of the dataset.
        """
        if self.mode == 'train':
            return len(self.data)
        else:
            return len(self.images)
