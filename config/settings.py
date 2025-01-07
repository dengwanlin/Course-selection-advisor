import os

MONGODB_CONFIG = {
    'database': os.getenv("MONGO_DB_NAME", "Course_Recommendation"),
}
