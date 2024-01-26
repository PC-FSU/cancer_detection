from src.cancer_detection.config.configuration import ConfigurationManager
from src.cancer_detection.components.data_ingestion import DataIngestion
from src.cancer_detection import logger

STAGE_NAME = "Data Ingestion stage"

class DataIngestionTrainingPipeline:
    def __init__(self):
        """
        Initializes the DataIngestionTrainingPipeline.
        """
        pass

    def main(self):
        """
        Main function to execute the Data Ingestion stage.

        - Initializes ConfigurationManager to retrieve configuration.
        - Gets DataIngestionConfig using ConfigurationManager.
        - Creates an instance of DataIngestion with the obtained configuration.
        - Downloads data and extracts the zip file using DataIngestion.
        """
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
