"""
Data utilities for the Car Price Prediction project.
"""

import logging

logger = logging.getLogger(__name__)




def load_data(spark, file_path):
    """
    Load data from a CSV file into a PySpark DataFrame.
    
    Args:
        spark: SparkSession object
        file_path (str): Path to the CSV file
        
    Returns:
        DataFrame: PySpark DataFrame with the loaded data
    """
    logger.info(f"Loading data from {file_path}")
    return spark.read.csv(file_path, header=True, inferSchema=True)






def save_as_csv(spark_df, file_path="output.csv"):
    """
    Convert a PySpark DataFrame to a Pandas DataFrame and save it as a CSV file.
    
    Args:
        spark_df: PySpark DataFrame to save
        file_path (str): Path where to save the CSV file
    """
    logger.info(f"Saving data to {file_path}")
    pandas_df = spark_df.toPandas()
    pandas_df.to_csv(file_path, index=False)





def drop_columns(df, columns):
    """
    Drop columns from a PySpark DataFrame.
    
    Args:
        df: PySpark DataFrame
        columns (list or str): Column(s) to drop
        
    Returns:
        DataFrame: PySpark DataFrame without the specified columns
    """
    if isinstance(columns, str):
        columns = [columns]
    
    # Only drop columns that exist in the DataFrame
    columns_to_drop = [col for col in columns if col in df.columns]
    
    if columns_to_drop:
        logger.info(f"Dropping columns: {columns_to_drop}")
        return df.drop(*columns_to_drop)
    return df






def drop_missing_values(df):
    """
    Handle missing values in a PySpark DataFrame.
    
    Args:
        df: PySpark DataFrame
        
    Returns:
        DataFrame: PySpark DataFrame with missing values handled
    """
    logger.info("Handling missing values")
    return df.dropna()



