import gdown
import torch

from src.cancer_detection.components.model import vgg16_modified, cancerClassifier
from src.cancer_detection.components.data_module import ImageTransform
from src.cancer_detection.entity.config_dataclasses import TrainingConfig, InfernceConfig
from src.cancer_detection.constants import DEVICE
from src.cancer_detection import logger
from PIL import Image
from typing import Type, List, Dict
from pathlib import Path

class PredictionPipeline:
    def __init__(self):
        """
        Initializes the PredictionPipeline.
        """
        pass

    @staticmethod
    def _predict(
        training_config: TrainingConfig,
        inference_config: InfernceConfig,
        image_transformation_pipeline: Type[ImageTransform],
        filename : Path
    ) -> Dict[str, str]:
        """
        Performs the prediction using the trained model.

        Args:
        - training_config (TrainingConfig): Configuration for training.
        - inference_config (InfernceConfig): Configuration for inference.
        - image_transformation_pipeline (Type[ImageTransform]): Image transformation pipeline.
        - filename (Path): Path to the image file for prediction.

        Returns:
        - Dict[str, str]: Prediction result in a dictionary.
        """
        
        # Load the model
        model = vgg16_modified(training_config)

        if inference_config.load_from_local:
            logger.info(f"Trying to load checkpoint from local dir : {inference_config.path_to_best_checkpoint_local}")
            # Load checkpoint locally
            try:
                model_ = cancerClassifier.load_from_checkpoint(inference_config.path_to_best_checkpoint_local, model=model, config=training_config)
            except FileNotFoundError:
                logger.info(f"Error: File '{inference_config.path_to_best_checkpoint_local}' not found.")
            except Exception as e:
                logger.info(f"Error: An unexpected error occurred - {e}")
        else:
            checkpoint_url = str(inference_config.URL_to_load_from_drive)
            out_file = str(inference_config.best_model_checkpoints_saved_from_URL)

            if not inference_config.best_model_checkpoints_saved_from_URL.is_file():
                # Download the checkpoint from Google Drive
                logger.info(f"Downloading data from {checkpoint_url} into file {out_file}")
                file_id = checkpoint_url.split("/")[-2]
                prefix = 'https://drive.google.com/uc?/export=download&id='
                gdown.download(prefix + file_id, out_file)
            else:
                logger.info(f"Using cached checkpoint from: {inference_config.best_model_checkpoints_saved_from_URL}")
                
            model_ = cancerClassifier.load_from_checkpoint(out_file, model=model, config=training_config)

        model_.eval()
        
        # Load image
        img = Image.open(filename)
        # Transform image
        test_image = image_transformation_pipeline(img, "test").unsqueeze(dim=0).to(DEVICE)

        # Make predictions
        logits = model_(test_image)
        preds = torch.argmax(logits, dim=1)

        if preds[0] == 1:
            prediction = 'Normal'
            return {"image": prediction}
        else:
            prediction = 'Adenocarcinoma Cancer'
            return {"image": prediction}
