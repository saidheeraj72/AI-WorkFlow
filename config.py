from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/chatbot_db")
    
def get_db():
    client = MongoClient(Config.MONGODB_URI)
    return client.get_default_database()