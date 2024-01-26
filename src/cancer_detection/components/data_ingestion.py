import os
import zipfile
import gdown
from src.cancer_detection import logger
from src.cancer_detection.utils.common import get_size
from src.cancer_detection.entity.config_dataclasses import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        """
        Constructor for DataIngestion class.

        Parameters:
        - config (DataIngestionConfig): Configuration object for data ingestion.

        Returns:
        None
        """
        self.config = config

    def download_file(self) -> str:
        '''
        Fetch data from the URL.

        Returns:
        str: Path to the downloaded file.
        '''
        try: 
            # Extracting configuration parameters
            dataset_url = self.config.source_URL
            zip_download_dir = self.config.local_data_file

            # Creating necessary directories
            os.makedirs("artifacts/data_ingestion", exist_ok=True)

            # Logging download information
            logger.info(f"Downloading data from {dataset_url} into file {zip_download_dir}")
            
            # Extracting file ID from the URL and constructing download URL
            file_id = dataset_url.split("/")[-2]
            prefix = 'https://drive.google.com/uc?/export=download&id='
            gdown.download(prefix + file_id, zip_download_dir)

            # Logging successful download
            logger.info(f"Downloaded data from {dataset_url} into file {zip_download_dir}")

            return zip_download_dir  # Returning the path to the downloaded file

        except Exception as e:
            # Raising an exception if download fails
            raise e
        
    
    def extract_zip_file(self):
        """
        Extracts the contents of the ZIP file into the data directory.

        Returns:
        None
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        
        # Extracting contents of the ZIP file
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
