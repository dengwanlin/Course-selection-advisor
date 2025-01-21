import os

MONGODB_CONFIG = {
    'host': os.getenv("MONGO_HOST", "localhost"),
    'port': int(os.getenv("MONGO_PORT", 27017)),
    'database': os.getenv("MONGO_DB_NAME", "course_recommendation"),
}
