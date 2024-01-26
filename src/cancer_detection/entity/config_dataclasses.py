from dataclasses import dataclass
from pathlib import Path

# Configuration for Data Ingestion
@dataclass
class DataIngestionConfig:
    root_dir: Path                # Root directory for data ingestion
    source_URL: str               # URL to the data source
    local_data_file: Path          # Local file to store downloaded data
    unzip_dir: Path               # Directory to extract the data

# Configuration for Training
@dataclass
class TrainingConfig:
    root_dir: Path                # Root directory for training
    training_data: Path           # Directory containing training data
    model_checkpoints: Path       # Directory to save model checkpoints
    params_is_augmentation: bool  # Flag indicating whether data augmentation is used
    params_image_size: list       # List specifying the input image size [height, width]
    params_batch_size: int         # Batch size for training
    params_epochs: int             # Number of training epochs
    params_num_classes: int        # Number of classes in the dataset
    params_learning_rate: float    # Learning rate for optimization

# Configuration for MLflow
@dataclass
class MlflowConfig:
    mlflow_tracking_uri: str       # URI for MLflow tracking server
    mlflow_username: str           # Username for MLflow authentication
    mlflow_password: str           # Password for MLflow authentication

# Configuration for Inference
@dataclass
class InfernceConfig:
    load_from_local: bool          # Flag indicating whether to load model locally
    path_to_best_checkpoint_local: Path  # Path to the best model checkpoint locally
    URL_to_load_from_drive: Path   # URL to load the model checkpoint from drive
    best_model_checkpoints_saved_from_URL: Path  # Path to save the model checkpoint from URL
    unzip_dir: Path                # Directory to extract the downloaded model checkpoint
