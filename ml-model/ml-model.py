from pyspark.sql import SparkSession
from pymongo import MongoClient
import pandas as pd
from db.connection import connect_to_cluster

def fetch_courses_data():
    #Fetch courses data from MongoDB and return as a Pandas DataFrame.

    # Connect to MongoDB cluster
    client = connect_to_cluster()
    
    # Access the database and collection
    db = client['Course_Recommendation']
    collection = db['course']
    
    # Fetch data from the collection
    courses_data = list(collection.find())
    
    # Convert to Pandas DataFrame
    df = pd.DataFrame(courses_data)    
    return df

def create_spark_session(app_name="CourseSelectionAdvisor"):
    """
    Create and return a Spark session.
    """
    spark = SparkSession.builder \
        .appName(app_name) \
        .getOrCreate()
    return spark

def load_data(spark, file_path):
    """
    Load data from a CSV file into a Spark DataFrame.
    """
    df = spark.read.csv(file_path, header=True, inferSchema=True)
    return df

def main():
    # Create Spark session
    # spark = create_spark_session()
    # Fetch courses data from MongoDB
    # courses_df = fetch_courses_data()

    # # Convert Pandas DataFrame to Spark DataFrame
    # spark_courses_df = spark.createDataFrame(courses_df)

    # # Show the first 5 rows of the DataFrame
    # spark_courses_df.show(5)

    # # Stop the Spark session
    # spark.stop()
    # Fetch courses data from MongoDB
    courses_df = fetch_courses_data()

    # Show the first 5 rows of the DataFrame
    print(courses_df.head(20))


  

if __name__ == "__main__":
    main()