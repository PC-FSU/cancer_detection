from src.cancer_detection.constants import *
from src.cancer_detection.utils.common import read_yaml, create_directories
from src.cancer_detection.entity.config_dataclasses import DataIngestionConfig, TrainingConfig, MlflowConfig, InfernceConfig
import os

class ConfigurationManager:
    def __init__(
            self,
            config_filepath=CONFIG_FILE_PATH,
            params_filepath=PARAMS_FILE_PATH
        ):
        """
        Initializes ConfigurationManager with file paths for configuration and parameters.

        Args:
        - config_filepath (str): File path for configuration YAML file.
        - params_filepath (str): File path for parameters YAML file.
        """
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        
        # Create the artifact model
        create_directories([self.config.artifacts_root])
    

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """
        Retrieves DataIngestionConfig from the configuration.

        Returns:
        - DataIngestionConfig: Configuration for data ingestion.
        """
        config = self.config.data_ingestion

        # Create a subfolder in artifact root folder, where data ingestion-related files will be put
        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
    
    
    def get_training_config(self) -> TrainingConfig:
        """
        Retrieves TrainingConfig from the configuration.

        Returns:
        - TrainingConfig: Configuration for training.
        """
        config = self.config.training
        params = self.params
        training_data = os.path.join(self.config.data_ingestion.unzip_dir, "Chest-CT-Scan-data")
        
        create_directories([Path(config.root_dir)])
        create_directories([Path(config.model_checkpoints)])

        training_config = TrainingConfig(
            root_dir=Path(config.root_dir),
            training_data=Path(training_data),
            model_checkpoints=config.model_checkpoints,
            params_is_augmentation=params.AUGMENTATION,
            params_image_size=params.IMAGE_SIZE,
            params_batch_size=params.BATCH_SIZE,
            params_epochs=params.EPOCHS,
            params_num_classes=params.CLASSES,
            params_learning_rate=params.LEARNING_RATE
        )

        return training_config


    def get_mlflow_config(self) -> MlflowConfig:
        """
        Retrieves MlflowConfig from the configuration.

        Returns:
        - MlflowConfig: Configuration for MLflow.
        """
        config = self.config.mlflow
        
        mlflow_config = MlflowConfig(
            mlflow_tracking_uri=str(config.MLFLOW_TRACKING_URI),
            mlflow_username=str(config.MLFLOW_TRACKING_USERNAME),
            mlflow_password=str(config.MLFLOW_TRACKING_PASSWORD)
        )
        
        return mlflow_config


    def get_inference_config(self) -> InfernceConfig:
        """
        Retrieves InfernceConfig from the configuration.

        Returns:
        - InfernceConfig: Configuration for inference.
        """
        config = self.config.inference
        
        create_directories([Path(config.root_dir)])
        create_directories([Path(config.unzip_dir)])

        inference_config = InfernceConfig(
            load_from_local=config.load_from_local,
            path_to_best_checkpoint_local=Path(config.path_to_best_checkpoint_local),
            URL_to_load_from_drive=Path(config.URL_to_load_from_drive),
            best_model_checkpoints_saved_from_URL=Path(config.best_model_checkpoints_saved_from_URL),
            unzip_dir=Path(config.unzip_dir)
        )
        
        return inference_config
