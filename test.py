from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

# Manually URL-encoded URI (@ becomes %40)
MONGODB_URI = "mongodb+srv://saidheeraj72:Sai%4017895@cluster0.yubdrhn.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

print(f"🔗 Testing with manual URI...")
print(f"URI (first 60 chars): {MONGODB_URI[:60]}...")

try:
    client = MongoClient(MONGODB_URI, server_api=ServerApi('1'))
    client.admin.command('ping')
    print("✅ SUCCESS: Connected to MongoDB!")
    
    # Quick database test
    db = client.ai_workflow
    print(f"📁 Database: {db.name}")
    print(f"📄 Collections: {db.list_collection_names()}")
    
except Exception as e:
    print(f"❌ FAILED: {e}")
