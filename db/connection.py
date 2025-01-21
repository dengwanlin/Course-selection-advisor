from pymongo import MongoClient
from pymongo.server_api import ServerApi
from config.settings import MONGODB_CONFIG

def get_connection():
    try:
        client = MongoClient(
            host=MONGODB_CONFIG['host'],
            port=MONGODB_CONFIG['port']
        )
        db = client[MONGODB_CONFIG['database']]
        return db
    except Exception as e:
        print("Error connecting to MongoDB:", e)
        return None

def connect_to_cluster():
    uri = "mongodb+srv://whhxsg:whhxsg@coursecluster.ecl2n.mongodb.net/?retryWrites=true&w=majority&appName=CourseCluster"

    # Create a new client and connect to the server
    client = MongoClient(uri, server_api=ServerApi('1'))

    # Send a ping to confirm a successful connection
    try:
        client.admin.command('ping')
        print("Pinged your deployment. You successfully connected to MongoDB!")
        return client
    except Exception as e:
        print(e)

def get_collections():
   client = connect_to_cluster()
   db = client['Course_Recommendation']

   collections = db.list_collection_names()
    # Print the collections
   print("List of collections")
   for coll in collections:
        print(coll)


def fetch_data(collection_name):
    client = connect_to_cluster()
    db = client['Course_Recommendation']
    collection = db[collection_name]
    X = collection.find()
    return list(X)

def fetch_local_data(collection_name):
    client = get_connection()
    db = client['course_recommendation']
    collection = db[collection_name]
    X = collection.find()
    return list(X)



