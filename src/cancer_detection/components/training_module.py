import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import os
import time
import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from src.cancer_detection import logger
from src.cancer_detection.entity.config_dataclasses import TrainingConfig, MlflowConfig
from src.cancer_detection.components.data_module import cancerDataModule
from src.cancer_detection.components.model import vgg16_modified, cancerClassifier

def train(config: TrainingConfig, mlflow_config: MlflowConfig, fast_dev_run: bool = False):
    """
    Run the full data-loading and model-training loop.

    Parameters:
    - config (TrainingConfig): Training configuration data class.
    - mlflow_config (MlflowConfig): Mlflow configuration data class.
    - fast_dev_run (bool): Flag for fast development run.

    Returns:
    None
    """

    # Set seed to control randomness
    seed = 123
    torch.manual_seed(seed)

    # Prepare data module
    num_workers = max(0, (os.cpu_count() or 1) - 1)
    datamodule = cancerDataModule(
        path=config.training_data,
        batch_size=config.params_batch_size,
        num_workers=num_workers,
        img_size=config.params_image_size[0]
    )

    # Saving the model checkpoint locally
    callbacks = [
        ModelCheckpoint(
            dirpath=config.model_checkpoints,
            filename="validation_{epoch}-{step}-{val_loss:.1f}",
            monitor="val_acc",
            save_top_k=1,  # save all checkpoints
            mode="max",
            every_n_epochs=5
        ),
        EarlyStopping(
            monitor="val_acc",
            mode="max",
            patience=20,
            verbose=True,
        ),
    ]

    # model
    model = vgg16_modified(config)
    # pytorch Trainer
    learner = cancerClassifier(model, config)
    trainer = pl.Trainer(
        max_epochs=config.params_epochs,
        fast_dev_run=fast_dev_run,
        enable_checkpointing=True,
        callbacks=callbacks
    )

    # add the mlflow vars to environment variable
    os.environ["MLFLOW_TRACKING_URI"] = mlflow_config.mlflow_tracking_uri
    os.environ["MLFLOW_TRACKING_USERNAME"] = mlflow_config.mlflow_username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = mlflow_config.mlflow_password

    # Tracking the model at remote dagshub
    client = MlflowClient()
    experiment_name = "vgg16classifier"
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_name = f"{timestamp}"

    try:
        experiment_id = client.create_experiment(experiment_name)
        experiment = client.get_experiment(experiment_id)
    except MlflowException:
        experiment = client.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id

    # Fetch experiment metadata information
    logger.info(f"Name: {experiment.name}")
    logger.info(f"Experiment_id: {experiment.experiment_id}")
    logger.info(f"Artifact Location: {experiment.artifact_location}")
    logger.info(f"Tags: {experiment.tags}")
    logger.info(f"Lifecycle_stage: {experiment.lifecycle_stage}")

    mlflow.set_tracking_uri(mlflow_config.mlflow_tracking_uri)
    # Activate auto logging for pytorch lightning module
    mlflow.pytorch.autolog(log_models=False)

    # Start MLflow run
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
        logger.info("tracking uri:", mlflow.get_tracking_uri())
        mlflow.log_params(config.__dict__)
        logger.info("Training model...")
        trainer.fit(learner, datamodule=datamodule)
