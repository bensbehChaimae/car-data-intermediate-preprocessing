"""
Main preprocessing pipeline for the Car Price Prediction project.
"""

from src.utils.data_utils import load_data , save_as_csv, drop_columns, drop_missing_values
from src.utils.spark_utils import  transform_spark_data
from src.utils.feature_utils import (
    split_equipment, extract_date_features, label_encode_categorical, 
    convert_boolean_to_int, convert_to_double, rename_columns_with_encoding_fix, fix_equipment_columns
    )

import logging


logger = logging.getLogger(__name__)


class Preprocessing_pipeline() :
    
    def __init__(self, spark, config):
        '''
        Initialize the processing pipeline.
        Args: 
            spark: SparkSession object
            config (dict): Configuration dictionary
        '''
        self.spark = spark
        self.config = config
        self.df = None 




    def run(self):
        '''
        Run the full preprocessing pipeline.
        '''
        logger.info("Starting preprocessing pipeline")

        # Step 1: Load and prepare the data
        self._prepare_data()

        # Step 2 : Perform feature engineering 
        self._feature_engineering()

        # Step 3: Encode variable
        self._variable_encoding()

        # Step 4: Save final processed data
        save_as_csv(self.df , self.config["paths"]["processed_data"])

        return self.df 
    

    def _prepare_data(self) :
        '''
        Load and prepare the raw data
        '''
        logger.info("Preparing the data for preprocessing")

        # Load raw data :
        raw_data = load_data(self.spark , self.config["paths"]["raw_data"])

        # Transform data
        data = transform_spark_data(raw_data)
        
        # Drop useless columns
        useless_columns = self.config["preprocessing"]["useless_columns"]
        data = drop_columns(data, useless_columns)
        
        self.df = data
        logger.info("Data preparation completed")
        
        return self.df
    






    def _feature_engineering(self):
        """
        Perform feature engineering on the data.
        """
        logger.info("Performing feature engineering")
        
        # Handle missing values
        data_cleaned = drop_missing_values(self.df)
        
        # Extract date features
        if "publication_date" in data_cleaned.columns:
            data_features = extract_date_features(data_cleaned)
            # Drop original date column
            data_features = drop_columns(data_features, "publication_date")
        else:
            data_features = data_cleaned
            logger.warning("publication_date column not found, skipping date feature extraction")
        
        # Split equipment column
        equipment_types = self.config["preprocessing"]["equipment_types"]
        if "equipment" in data_features.columns:
            data_equipment = split_equipment(data_features, equipment_types)
            # Drop original equipment column
            data_equipment = drop_columns(data_equipment, "equipment")
        else:
            data_equipment = data_features
            logger.warning("equipment column not found, skipping equipment splitting")
        
        # Rename equipment columns - standard approach for non-problematic columns
        equipment_mapping = self.config["preprocessing"]["equipment_mapping"]
        data_equipment = rename_columns_with_encoding_fix(data_equipment, equipment_mapping)
        
        # Direct fix for problematic columns with encoding issues
        data_equipment = fix_equipment_columns(data_equipment)
        
        # Save intermediate result
        save_as_csv(data_equipment, self.config["paths"]["interim_data"]["feature_engineering"])
        
        self.df = data_equipment
        logger.info("Feature engineering completed")
        
        return self.df
    


    
    def _variable_encoding(self):
        """
        Encode variables for machine learning.
        """
        logger.info("Encoding variables")
        
        # Get column lists from config
        bool_cols = self.config["preprocessing"]["column_types"]["boolean_columns"]
        numeric_cols = self.config["preprocessing"]["column_types"]["numeric_columns"]
        categorical_cols = self.config["preprocessing"]["column_types"]["categorical_columns"]
        
        # Convert boolean columns to integer
        data_encoded = convert_boolean_to_int(self.df, bool_cols)
        
        # Convert numeric columns to double
        data_encoded = convert_to_double(data_encoded, numeric_cols)
        
        # Label encode categorical columns
        data_encoded = label_encode_categorical(data_encoded, categorical_cols)
        
        # Save intermediate result
        save_as_csv(data_encoded, self.config["paths"]["interim_data"]["variable_encoding"])
        
        self.df = data_encoded
        logger.info("Variable encoding completed")
        
        return self.df



 

