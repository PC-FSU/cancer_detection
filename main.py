from src.cancer_detection.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.cancer_detection.pipeline.stage_02_training_and_tracking import TrainingPipeline
from src.cancer_detection import logger

# Stage 01: Data Ingestion
STAGE_NAME = "Data Ingestion stage"

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    # Create an instance of the DataIngestionTrainingPipeline
    obj = DataIngestionTrainingPipeline()
    # Execute the main method to perform data ingestion
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    # Log any exceptions that occur during data ingestion
    logger.exception(e)
    raise e


# Stage 02: Training Pipeline
STAGE_NAME = "TRAINING PIPELINE"
try:
    logger.info(f"*******************")
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    # Create an instance of the TrainingPipeline
    obj = TrainingPipeline()
    # Execute the main method to perform training
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    # Log any exceptions that occur during training
    logger.exception(e)
    raise e
