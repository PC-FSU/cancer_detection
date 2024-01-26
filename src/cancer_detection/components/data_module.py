import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from typing import Dict, List, Optional
import torchvision.transforms as transforms
import pytorch_lightning as pl
import glob
import numpy as np
import os
from src.cancer_detection import logger

# Data Augmentation
class ImageTransform():
    def __init__(
            self,
            img_size: int = 224,
            mean: Optional[list] = None,
            std: Optional[list] = None
    ):
        """
        Image transformation class for data augmentation.

        Parameters:
        - img_size (int): Size of the transformed images.
        - mean (Optional[list]): Mean values for normalization (default is ImageNet mean).
        - std (Optional[list]): Standard deviation values for normalization (default is ImageNet std).

        Returns:
        None
        """
        if mean is None:
            self.mean = [0.485, 0.456, 0.406]
        if std is None:
            self.std = [0.229, 0.224, 0.225]

        self.data_transform = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(img_size),
                transforms.ToTensor()
            ]),
            'test': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(img_size),
                transforms.ToTensor()
            ])
        }

    def __call__(self, img: Image.Image, phase: str) -> torch.tensor:
        """
        Applies data augmentation transformations to the input image.

        Parameters:
        - img (Image.Image): Input image.
        - phase (str): Current phase ('train', 'val', or 'test').

        Returns:
        torch.tensor: Transformed image tensor.
        """
        img = self.data_transform[phase](img)

        if img.shape[0] == 1:
            logger.info(f'logging shape : \n {img.shape}, dtype : {img.dtype}')
            img = torch.repeat_interleave(img, 3, dim=0)
        img = img[:3, :, :]

        # Normalize
        img = transforms.functional.normalize(img, mean=self.mean, std=self.std)
        return img


class cancerDataset(Dataset):
    """A PyTorch Dataset for the cancer image data."""
    def __init__(self, file_list, transform_fun=None, phase='train') -> None:
        """
        Constructor for cancerDataset class.

        Parameters:
        - file_list (List): List of file paths for the dataset.
        - transform_fun: Transformation function for data augmentation.
        - phase (str): Current phase ('train', 'val', or 'test').

        Returns:
        None
        """
        self.file_list = file_list
        self.transform = transform_fun
        self.phase = phase

    def __len__(self) -> int:
        """Returns the number of examples in the dataset."""
        return len(self.file_list)

    def __getitem__(self, idx: int):
        """
        Returns the transformed image tensor and label for the given index.

        Parameters:
        - idx (int): Index of the data sample.

        Returns:
        Tuple[torch.tensor, int]: Transformed image tensor and corresponding label.
        """
        img_path = self.file_list[idx]
        img = Image.open(img_path)

        # Transforming Image
        img_transformed = self.transform(img, self.phase)

        # Get Label
        label = img_path.split("/")[-2]
        if label == 'adenocarcinoma':
            label = 0
        elif label == 'normal':
            label = 1

        return img_transformed, label


class cancerDataModule(pl.LightningDataModule):
    def __init__(self, path, img_size, batch_size, num_workers, seed: Optional[int] = None) -> None:
        """
        Constructor for cancerDataModule class.

        Parameters:
        - path (str): Root path of the dataset.
        - img_size (int): Size of the transformed images.
        - batch_size (int): Batch size for DataLoader.
        - num_workers (int): Number of DataLoader workers.
        - seed (Optional[int]): Random seed for reproducibility.

        Returns:
        None
        """
        super().__init__()
        self.path = path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.img_size = img_size
        self.rng = np.random.default_rng(seed)

    def setup(self, stage: str) -> None:
        """
        Splits the dataset into training, validation, and test sets.

        Parameters:
        - stage (str): Current stage ('fit' or 'test').

        Returns:
        None
        """
        self.files = np.array(glob.glob(os.path.join(self.path, '**/*.png'), recursive=True))
        train_size = int(0.8 * len(self.files))
        val_size = int(0.1 * len(self.files))
        indices = np.arange(len(self.files))
        self.rng.shuffle(indices)
        train_indices = indices[:train_size]
        val_indices = indices[train_size: train_size + val_size]
        test_indices = indices[train_size + val_size:]
        self.train_files = self.files[train_indices]
        self.val_files = self.files[val_indices]
        self.test_files = self.files[test_indices]

    def train_dataloader(self) -> DataLoader:
        """Returns a DataLoader for the training set."""
        dataset = cancerDataset(self.train_files, ImageTransform(self.img_size), 'train')
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        """Returns a DataLoader for the validation set."""
        dataset = cancerDataset(self.val_files, ImageTransform(self.img_size), 'val')
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        """Returns a DataLoader for the test set."""
        dataset = cancerDataset(self.test_files, ImageTransform(self.img_size), 'test')
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self) -> DataLoader:
        """Returns a DataLoader for prediction."""
        dataset = cancerDataset(self.test_files)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )