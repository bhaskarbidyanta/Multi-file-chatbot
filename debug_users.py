import pymongo
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# MongoDB connection
MONGO_URI = os.getenv("MONGO_URI")
client = pymongo.MongoClient(MONGO_URI)
db = client["chatbot_db"]
users_collection = db["users"]

print("Users in the database:")
for user in users_collection.find():
    print(user)
