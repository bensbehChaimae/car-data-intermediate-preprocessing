'''
Spark utils for the project 
'''

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when





# Function to create and configure spark session : 
def create_spark_session(app_name = "Car price prediction",
                         driver_memory = "4g", executor_memory="4g") :
    
    '''
    Args:
        app_name (str) : name of Spark application.
        driver_memory (str) : amout of memory to allocate to the driver.
        executor_memory (str) : amout of memory to allocate to each executor. 

    Returns:
        SparkSession: Configured Spark session.
    '''

    spark = SparkSession.builder \
        .appName(app_name) \
        .config("spark.driver.memory" , driver_memory) \
        .config("spark.executor.memory" , executor_memory) \
        .getOrCreate()
    
    return spark









def transform_spark_data(df):
    
    """
    Transform data in a PySpark DataFrame, handling NaN values.
    
    Args:
        df (DataFrame): PySpark DataFrame to transform
        
    Returns:
        DataFrame: Transformed DataFrame
    """
    
    for column, dtype in df.dtypes:
        if dtype == 'string':
            df = df.withColumn(
                column,
                when(
                    col(column).rlike(r'(?i)\\\"nan\\\"'),
                    None
                ).otherwise(col(column))
            )
    return df






