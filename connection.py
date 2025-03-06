from pymongo import MongoClient
import os
mongo_uri = os.getenv("MONGO_URI")
client = MongoClient(mongo_uri, serverSelectionTimeoutMS=10000)  # 5-second timeout

try:
    client.admin.command("ping")  # Check if connected
    print("Connected to MongoDB!")
except Exception as e:
    print(f"Erstrror: {e}")
