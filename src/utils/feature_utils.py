"""
Feature engineering utilities for the Car Price Prediction project.
"""

import logging
from pyspark.sql import functions as F
from pyspark.sql.functions import col, when, lower, lit
from pyspark.sql.types import IntegerType
from pyspark.ml.feature import StringIndexer

logger = logging.getLogger(__name__)





def split_equipment(df, equipment_types, equipment_col="equipment"):
    """
    Split equipment column into multiple boolean columns with ASCII-only names.
    
    Args:
        df: PySpark DataFrame
        equipment_types (list): List of equipment types to extract
        equipment_col (str): Name of the equipment column
        
    Returns:
        DataFrame: PySpark DataFrame with equipment split into boolean columns
    """
    logger.info("Splitting equipment column")
    
    # Check if equipment column exists
    if equipment_col not in df.columns:
        logger.warning(f"Equipment column '{equipment_col}' not found in DataFrame")
        return df
    
    for eq in equipment_types:
        # Generate a clean column name without accents
        import unicodedata
        
        # Normalize to ASCII
        eq_normalized = unicodedata.normalize('NFKD', eq)
        eq_ascii = ''.join([c for c in eq_normalized if not unicodedata.combining(c)])
        
        # Generate clean column name
        new_col = eq_ascii.lower().replace(" ", "_").replace("/", "_").replace("-", "_")
        
        # Add column with True/False if the equipment exists
        df = df.withColumn(
                new_col,
                when(lower(col(equipment_col)).contains(eq.lower()), lit(True)).otherwise(lit(False))
            )
    
    return df








def extract_date_features(df, date_col="publication_date", date_format="dd/MM/yyyy HH:mm"):
    """
    Extract features from a date column.
    
    Args:
        df: PySpark DataFrame
        date_col (str): Name of the date column
        date_format (str): Format of the date
        
    Returns:
        DataFrame: PySpark DataFrame with extracted date features
    """
    logger.info(f"Extracting date features from {date_col}")
    
    # Check if date column exists
    if date_col not in df.columns:
        logger.warning(f"Date column '{date_col}' not found in DataFrame")
        return df
    
    # Convert to timestamp
    df = df.withColumn(date_col, F.to_timestamp(date_col, date_format))
    
    # Extract features
    df = df.withColumn("publication_Year", F.year(date_col)) \
        .withColumn("publication_Month", F.month(date_col)) \
        .withColumn("publication_Day", F.dayofmonth(date_col)) \
        .withColumn("publication_Weekday", F.dayofweek(date_col)) \
        .withColumn("Is_Weekend", (F.dayofweek(date_col) >= 6).cast(IntegerType())) \
        .withColumn("Days_since_posted", F.datediff(F.current_date(), date_col))
    
    return df









def label_encode_categorical(df, columns):
    """
    Perform label encoding on categorical columns.
    
    Args:
        df: PySpark DataFrame
        columns (list): List of categorical columns to encode
        
    Returns:
        DataFrame: PySpark DataFrame with encoded categorical columns
    """
    logger.info(f"Label encoding columns: {columns}")
    
    # Filter out columns that don't exist in the DataFrame
    columns = [col for col in columns if col in df.columns]
    
    for col_name in columns:
        indexer = StringIndexer(inputCol=col_name, outputCol=col_name + "_idx")
        df = indexer.fit(df).transform(df).drop(col_name).withColumnRenamed(col_name + "_idx", col_name)
    
    return df






def convert_boolean_to_int(df, columns):
    """
    Convert boolean columns to integer type (0/1).
    
    Args:
        df: PySpark DataFrame
        columns (list): List of boolean columns to convert
        
    Returns:
        DataFrame: PySpark DataFrame with converted boolean columns
    """
    logger.info("Converting boolean columns to integers")
    
    # Filter out columns that don't exist in the DataFrame
    columns = [col_name for col_name in columns if col_name in df.columns]
    
    for col_name in columns:
        df = df.withColumn(col_name, col(col_name).cast(IntegerType()))
    
    return df









def convert_to_double(df, columns):
    """
    Convert columns to double type.
    
    Args:
        df: PySpark DataFrame
        columns (list): List of columns to convert
        
    Returns:
        DataFrame: PySpark DataFrame with converted columns
    """
    logger.info("Converting columns to double type")
    
    # Filter out columns that don't exist in the DataFrame
    columns = [col_name for col_name in columns if col_name in df.columns]
    
    for col_name in columns:
        df = df.withColumn(col_name, col(col_name).cast("double"))
    
    return df



def rename_columns_with_encoding_fix(df, mapping):
    """
    Rename columns according to a mapping, handling encoding issues with accented characters.
    
    Args:
        df: PySpark DataFrame
        mapping (dict): Dictionary mapping old column names to new ones
        
    Returns:
        DataFrame: PySpark DataFrame with renamed columns
    """
    logger.info("Renaming columns with encoding fix")
    
    # Get current columns
    current_columns = df.columns
    
    # For each target column in the mapping, look for a match or partial match
    for old_name, new_name in mapping.items():
        # First try exact match
        if old_name in current_columns:
            df = df.withColumnRenamed(old_name, new_name)
            continue
            
        # If no exact match, try to find a close match based on non-accented characters
        old_name_base = ''.join(c.lower() for c in old_name if c.isalnum())
        
        for col in current_columns:
            col_base = ''.join(c.lower() for c in col if c.isalnum())
            
            # If we find a reasonable match (80% of characters match)
            if old_name_base in col_base or col_base in old_name_base:
                logger.info(f"Found partial match: '{col}' -> '{new_name}' (was looking for '{old_name}')")
                df = df.withColumnRenamed(col, new_name)
                break
    
    return df





def fix_equipment_columns(df):
    """
    Directly fix problematic equipment column names with hardcoded mapping.
    
    Args:
        df: PySpark DataFrame
        
    Returns:
        DataFrame: PySpark DataFrame with fixed column names
    """
    logger.info("Fixing problematic equipment column names")
    
    # Direct mapping for problematic columns - add all columns that need renaming
    problem_columns = {
        "sia_ges_cuir": "Leather_seats",
        "cama©ra_de_recul": "Rear_camera",
        "vitres_a©lectriques": "Electric_windows",
        "ra©gulateur_de_vitesse": "Cruise_control",
        "verrouillage_centralisa©": "Central_locking"
    }
    
    # Rename each problematic column
    for old_col, new_col in problem_columns.items():
        if old_col in df.columns:
            df = df.withColumnRenamed(old_col, new_col)
            logger.info(f"Renamed '{old_col}' to '{new_col}'")
    
    return df


    
     





