from config import get_db

db = get_db()

# Create indexes for better performance
db.chats.create_index([("user_id", 1)])
db.chats.create_index([("created_at", -1)])