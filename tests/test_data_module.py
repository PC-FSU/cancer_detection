# Importing necessary libraries
import pytest
import torch
import glob
import sys
import os
from torch.utils.data import DataLoader
from PIL import Image
from pathlib import Path

sys.path.insert(0, os.path.join(Path(__file__).parent.parent))
print(sys.path)
from src.cancer_detection.components.data_module import ImageTransform, cancerDataset, cancerDataModule



# Fixture to create a temporary directory with dummy images for testing
@pytest.fixture
def file_list():
    file_list_ = glob.glob(os.path.join(os.getcwd(), "tests", "temp_dataset", "**/*.png"), recursive=True)
    return file_list_


# Test for ImageTransform class
def test_image_transform():
    # Creating an instance of ImageTransform
    transform = ImageTransform(img_size=224)
    # Creating a dummy image
    dummy_image = Image.new("RGB", (224, 224), color="white")

    # Applying the data transformation
    transformed_image = transform(dummy_image, phase="train")
    # Asserting the output is a tensor and has the correct shape
    assert isinstance(transformed_image, torch.Tensor)
    assert transformed_image.shape == (3, 224, 224)


# Test for cancerDataset class
def test_cancer_dataset(file_list):

    # Creating an instance of cancerDataset
    dataset = cancerDataset(file_list, transform_fun=ImageTransform(224), phase="train")

    # Asserting the length of the dataset is correct
    assert len(dataset) == 4

    # Obtaining a sample from the dataset
    sample = dataset[0]
    img, label = sample
    # Asserting the image is a tensor and the label is either 0 or 1
    assert isinstance(img, torch.Tensor)
    assert label in [0, 1]


# Test for cancerDataModule class
def test_cancer_data_module():
    # Setting up parameters for cancerDataModule
    img_size = 224
    batch_size = 2
    num_workers = 1
    seed = 42
    dummy_image_folder = os.path.join(os.getcwd(), "tests")
    # Creating an instance of cancerDataModule
    data_module = cancerDataModule(
        path=str(dummy_image_folder),
        img_size=img_size,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed
    )

    # Setting up the data module for the 'fit' stage
    data_module.setup("fit")

    # Obtaining DataLoader instances for training, validation, test, and prediction
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()
    predict_loader = data_module.predict_dataloader()

    # Asserting the DataLoader instances are of the correct type
    assert isinstance(train_loader, DataLoader)
    assert isinstance(val_loader, DataLoader)
    assert isinstance(test_loader, DataLoader)
    assert isinstance(predict_loader, DataLoader)

    # Asserting the datasets in the DataLoader instances are not empty
    assert len(train_loader.dataset) > 0