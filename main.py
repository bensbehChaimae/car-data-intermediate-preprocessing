"""
Main entry point for running the Car Price Prediction preprocessing pipeline.
"""

import os 
import yaml 
import logging
from pathlib import Path


from src.utils.spark_utils import create_spark_session
from src.preprocessing.pipeline import Preprocessing_pipeline


logging.basicConfig(
    level= logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


logger = logging.getLogger(__name__)


def load_config(config_path="config/config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config



def main():
    '''
    Main function to run the preprocessing pipeline
    '''
    logging.info("Loading configuration ....")
    config = load_config()

    # Create directories if they don't exist
    Path(os.path.dirname(config["paths"]["interim_data"]["feature_engineering"])).mkdir(parents=True, exist_ok=True)
    Path(os.path.dirname(config["paths"]["processed_data"])).mkdir(parents=True, exist_ok=True)


    # Create Spark Session :
    logging.info("Creating spark session ....")
    spark = create_spark_session(
        app_name= config["spark"]["app_name"],
        driver_memory=config["spark"]["driver_memory"],
        executor_memory=config["spark"]["executor_memory"]
    )


    # Initialize and run the preprocessing pipeline :
    logging.info("Initialize preprocessing pipeline ....")
    pipeline = Preprocessing_pipeline(spark, config)

    # Run the pipeline
    logger.info("Running preprocessing pipeline...")
    pipeline.run()
    
    logger.info("Preprocessing completed successfully!")
    
    # Stop Spark session
    spark.stop()




if __name__ == "__main__":
    main()
