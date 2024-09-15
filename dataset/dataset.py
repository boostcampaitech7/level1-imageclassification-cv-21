from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import cv2
import os

class CustomDataset(Dataset):
    def __init__(self, data_path, mode='train', transform=None):
        """
        Initializes the CustomDataset class.

        Args:
        - data_path (str): The path to the dataset.
        - mode (str): The mode of the dataset, either 'train' or 'test' (default: 'train').
        - transform (transforms.Compose): The transforms to be applied to the images (default: None).
        """
        self.data_path = os.path.join(data_path, f'{mode}')
        self.mode = mode
        self.data = pd.read_csv(os.path.join(data_path, f'{mode}.csv'))
        self.image_paths = self.data['image_path'].tolist()
        if mode == 'train':
            self.labels = self.data['target'].tolist()  # Read image paths from test.csv
        self.transform = transform

    
    def _load_image(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # 이미지를 BGR 컬러 포맷의 numpy array로 읽어옵니다.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR 포맷을 RGB 포맷으로 변환합니다.
        # image = self.transform(image)  # 설정된 이미지 변환을 적용합니다.
        image = Image.fromarray(image)
        return image
    
    def _apply_transform(self, image):
        if self.transform:
            image = self.transform(image)
        return image
    
    def __getitem__(self, index):
        """
        Returns the image and label at the specified index.

        Args:
        - index (int): The index of the image and label.

        Returns:
        - image: The image at the specified index.
        - label: The label at the specified index (Not returned if in 'test' mode).
        """
        image_path = os.path.join(self.data_path, self.image_paths[index])
        image = self._load_image(image_path)
        image = self._apply_transform(image)

        if self.mode != 'train':
            return image
        else:
            label = self.labels[index]           
            return image, label

    
    def __len__(self):
        return len(self.image_paths)
