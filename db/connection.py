from pymongo import MongoClient
from pymongo.server_api import ServerApi
import os
from dotenv import load_dotenv


# Load variables from .env file
load_dotenv()


# Access environment variables
db_host = os.getenv("DB_HOST")
db_port = os.getenv("DB_PORT")
local_db_name = os.getenv("LOCAL_DB_NAME")
db_name = os.getenv("DB_NAME")
db_uri = os.getenv("DB_URI")

def get_connection():
    try:
        client = MongoClient(
            host=db_host,
            port=db_port
        )
        db = client[local_db_name]
        return db
    except Exception as e:
        print("Error connecting to MongoDB:", e)
        return None

def connect_to_cluster():


    # Create a new client and connect to the server
    client = MongoClient(db_uri, server_api=ServerApi('1'))

    # Send a ping to confirm a successful connection
    try:
        client.admin.command('ping')
        print("Pinged your deployment. You successfully connected to MongoDB!")
        return client
    except Exception as e:
        print(e)

def get_dbname():
    return db_name
