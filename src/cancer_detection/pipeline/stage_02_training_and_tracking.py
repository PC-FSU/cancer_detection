from src.cancer_detection.config.configuration import ConfigurationManager
from src.cancer_detection.components.training_module import train
from src.cancer_detection import logger

STAGE_NAME = "TRAINING and MLFLOW TRACKING PIPELINE"

class TrainingPipeline:
    def __init__(self):
        """
        Initializes the TrainingPipeline.
        """
        pass
    
    def main(self):
        """
        Main function to execute the Training and MLFlow Tracking stage.

        - Initializes ConfigurationManager to retrieve configuration.
        - Gets TrainingConfig and MlflowConfig using ConfigurationManager.
        - Calls the train function with training configuration and MLFlow configuration.
        """
        try:
            config = ConfigurationManager()
            training_config = config.get_training_config()
            mlflow_config = config.get_mlflow_config()
            train(training_config, mlflow_config)
        except Exception as e:
            raise e

if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = TrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
