# auth.py
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer
from typing import Optional, Dict
import jwt
from datetime import datetime, timedelta
import motor.motor_asyncio
from bson import ObjectId
import os

# JWT Configuration
JWT_SECRET = os.getenv("JWT_SECRET", "your-secret-key-here-change-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24

# Admin Email Domains and Specific Emails
ADMIN_EMAIL_DOMAINS = [
    "iiitdm.ac.in"
]

ADMIN_EMAILS = [
    "cs21b2021@iiitdm.ac.in"
    # Add your specific admin emails here
]

security = HTTPBearer()

class AuthManager:
    def __init__(self, db):
        self.db = db
        self.users_collection = db.users

    def create_access_token(self, data: dict) -> str:
        """Create a JWT access token."""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS)
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)
        return encoded_jwt

    def verify_token(self, token: str) -> Dict:
        """Verify and decode JWT token."""
        try:
            payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
            email: str = payload.get("sub")
            if email is None:
                raise HTTPException(status_code=401, detail="Invalid token")
            return {"email": email}
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token has expired")
        except jwt.PyJWTError:
            raise HTTPException(status_code=401, detail="Invalid token")

    def determine_user_role(self, email: str) -> str:
        """Determine user role based on email."""
        # Check specific admin emails first
        if email in ADMIN_EMAILS:
            return "admin"
        
        # Check admin domains
        email_domain = email.split("@")[-1] if "@" in email else ""
        if email_domain in ADMIN_EMAIL_DOMAINS:
            return "admin"
        
        return "user"

    async def get_or_create_user(self, email: str) -> Dict:
        """Get existing user or create new one."""
        user = await self.users_collection.find_one({"email": email})
        
        if not user:
            # Auto-register user
            role = self.determine_user_role(email)
            user_data = {
                "_id": ObjectId(),
                "email": email,
                "role": role,
                "created_at": datetime.utcnow(),
                "last_login": datetime.utcnow(),
                "login_count": 1
            }
            result = await self.users_collection.insert_one(user_data)
            user = await self.users_collection.find_one({"_id": result.inserted_id})
        else:
            # Update last login
            await self.users_collection.update_one(
                {"_id": user["_id"]},
                {
                    "$set": {"last_login": datetime.utcnow()},
                    "$inc": {"login_count": 1}
                }
            )
        
        return {
            "id": str(user["_id"]),
            "email": user["email"],
            "role": user["role"]
        }

    async def get_current_user(self, credentials) -> Dict:
        """Get current authenticated user from JWT token."""
        try:
            token_data = self.verify_token(credentials.credentials)
            user = await self.users_collection.find_one({"email": token_data["email"]})
            
            if user is None:
                raise HTTPException(status_code=401, detail="User not found")
            
            return {
                "id": str(user["_id"]),
                "email": user["email"],
                "role": user["role"]
            }
        except Exception as e:
            raise HTTPException(status_code=401, detail=str(e))

    async def require_admin(self, current_user: Dict) -> Dict:
        """Ensure the current user is an admin."""
        if current_user["role"] != "admin":
            raise HTTPException(status_code=403, detail="Admin access required")
        return current_user

# Session and Document Access Control
class AccessControl:
    def __init__(self, db):
        self.db = db
        self.sessions_collection = db.sessions
        self.documents_collection = db.documents

    async def check_session_access(self, session_id: str, user_id: str) -> bool:
        """Check if user has access to session."""
        session = await self.sessions_collection.find_one({"_id": ObjectId(session_id)})
        if not session:
            return False
        return session.get("user_id") == user_id

    async def check_document_access(self, document_id: str, user_id: str, user_role: str) -> bool:
        """Check if user has access to document."""
        document = await self.documents_collection.find_one({"_id": ObjectId(document_id)})
        if not document:
            return False
        
        # Admin can access all documents
        if user_role == "admin":
            return True
        
        # Users can access admin documents (read-only) and their own chat documents
        if document.get("document_type") == "admin":
            return True
        elif document.get("document_type") == "chat":
            return document.get("user_id") == user_id
        
        return False

    async def filter_user_accessible_documents(self, user_id: str, user_role: str, document_ids: list = None) -> list:
        """Filter documents based on user access."""
        query = {}
        
        if user_role == "admin":
            # Admin sees all admin documents
            query = {"document_type": "admin"}
        else:
            # Users see admin documents + their own chat documents
            query = {
                "$or": [
                    {"document_type": "admin"},
                    {"document_type": "chat", "user_id": user_id}
                ]
            }
        
        if document_ids:
            query["_id"] = {"$in": [ObjectId(doc_id) for doc_id in document_ids]}
        
        documents = await self.documents_collection.find(query).to_list(1000)
        return [str(doc["_id"]) for doc in documents]

# Rate Limiting (Optional Enhancement)
class RateLimiter:
    def __init__(self):
        self.requests = {}
    
    def check_rate_limit(self, user_id: str, limit: int = 100, window: int = 3600) -> bool:
        """Simple in-memory rate limiting."""
        now = datetime.utcnow().timestamp()
        
        if user_id not in self.requests:
            self.requests[user_id] = []
        
        # Clean old requests
        self.requests[user_id] = [
            req_time for req_time in self.requests[user_id] 
            if now - req_time < window
        ]
        
        # Check limit
        if len(self.requests[user_id]) >= limit:
            return False
        
        # Add current request
        self.requests[user_id].append(now)
        return True

# Usage in main.py:
# from auth import AuthManager, AccessControl, RateLimiter
# 
# auth_manager = AuthManager(db)
# access_control = AccessControl(db)
# rate_limiter = RateLimiter()
# 
# async def get_current_user(credentials = Depends(security)):
#     return await auth_manager.get_current_user(credentials)
# 
# async def get_admin_user(current_user: dict = Depends(get_current_user)):
#     return await auth_manager.require_admin(current_user)
