from fastapi import FastAPI, HTTPException, UploadFile, File, Depends,Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import motor.motor_asyncio
from bson import ObjectId
import os
import base64
import io
import PyPDF2
import docx
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
from dotenv import load_dotenv
from groq import Groq, APIStatusError
import jwt
import hashlib

# Load environment variables from .env file
load_dotenv()

app = FastAPI(title="AI Workflow Backend", version="1.0.0")

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MongoDB Configuration ---
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017/chatbot_db")
client_db = motor.motor_asyncio.AsyncIOMotorClient(MONGODB_URL)
db = client_db.chatbot_db
sessions_collection = db.sessions
settings_collection = db.settings
documents_collection = db.documents
chat_documents_collection = db.chat_documents
users_collection = db.users
folders_collection = db.folders

# --- Authentication Configuration ---
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24
security = HTTPBearer()

# --- Groq API Configuration ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if GROQ_API_KEY:
    groq_client = Groq(api_key=GROQ_API_KEY)
else:
    groq_client = None
    print("Warning: GROQ_API_KEY not found. Groq API calls will fail.")

# --- Sentence Transformer for Embeddings ---
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# --- FAISS Index (Admin documents only) ---
admin_faiss_index: Optional[faiss.IndexFlatIP] = None
admin_document_texts: List[str] = []
admin_document_metadata: List[Dict] = []

FAISS_DATA_DIR = "faiss_data"
ADMIN_INDEX_FILE = os.path.join(FAISS_DATA_DIR, "admin_documents.faiss")
ADMIN_TEXTS_FILE = os.path.join(FAISS_DATA_DIR, "admin_texts.pkl")
ADMIN_METADATA_FILE = os.path.join(FAISS_DATA_DIR, "admin_metadata.pkl")

os.makedirs("uploads", exist_ok=True)
os.makedirs(FAISS_DATA_DIR, exist_ok=True)

# --- Authentication Helper Functions ---
def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return encoded_jwt

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        token = credentials.credentials
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")
        return user_id
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")

async def get_current_user(user_id: str = Depends(verify_token)):
    user = await users_collection.find_one({"_id": ObjectId(user_id)})
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

# --- Pydantic Models ---
class Message(BaseModel):
    role: str
    content: str
    timestamp: Optional[str] = None
    image_url: Optional[str] = None
    chat_documents: Optional[List[str]] = None
    references: Optional[List[Dict]] = None

class ChatRequest(BaseModel):
    message: str
    model: str
    session_id: Optional[str] = None
    image_data: Optional[str] = None
    selected_document_ids: Optional[List[str]] = None
    selected_folder_ids: Optional[List[str]] = None  # NEW: Add folder selection
    document_data: Optional[str] = None
    document_filename: Optional[str] = None
    action_type: Optional[str] = "chat"

class SessionResponse(BaseModel):
    id: str
    title: str
    model: str
    messages: List[Message]

class SessionSummary(BaseModel):
    id: str
    title: str
    message_count: int

class DocumentInfo(BaseModel):
    id: str
    name: str
    size: int
    type: str
    upload_date: datetime
    uploaded_by: str
    folder_id: Optional[str] = None
    folder_path: Optional[str] = None

class FolderCreate(BaseModel):
    name: str
    parent_id: Optional[str] = None

class FolderInfo(BaseModel):
    id: str
    name: str
    parent_id: Optional[str]
    created_date: datetime
    item_count: int
    path: Optional[str] = None

class User(BaseModel):
    email: str
    role: str = "user"

class LoginRequest(BaseModel):
    username: str
    password: str

class DocumentPreviewRequest(BaseModel):
    document_id: str

# --- Helper Functions for Admin Documents (FAISS) ---
def load_admin_faiss_index():
    global admin_faiss_index, admin_document_texts, admin_document_metadata
    try:
        if (os.path.exists(ADMIN_INDEX_FILE) and 
            os.path.exists(ADMIN_TEXTS_FILE) and 
            os.path.exists(ADMIN_METADATA_FILE)):
            admin_faiss_index = faiss.read_index(ADMIN_INDEX_FILE)
            with open(ADMIN_TEXTS_FILE, "rb") as f:
                admin_document_texts = pickle.load(f)
            with open(ADMIN_METADATA_FILE, "rb") as f:
                admin_document_metadata = pickle.load(f)
            print(f"Loaded admin FAISS index with {len(admin_document_texts)} chunks.")
        else:
            raise FileNotFoundError("Admin FAISS index files not found.")
    except Exception as e:
        print(f"Could not load admin FAISS index: {e}. Creating a new one.")
        admin_faiss_index = faiss.IndexFlatIP(384)
        admin_document_texts = []
        admin_document_metadata = []

def save_admin_faiss_index():
    try:
        os.makedirs(os.path.dirname(ADMIN_INDEX_FILE), exist_ok=True)
        if admin_faiss_index:
            faiss.write_index(admin_faiss_index, ADMIN_INDEX_FILE)
            with open(ADMIN_TEXTS_FILE, "wb") as f:
                pickle.dump(admin_document_texts, f)
            with open(ADMIN_METADATA_FILE, "wb") as f:
                pickle.dump(admin_document_metadata, f)
            print("Admin FAISS index saved successfully.")
    except Exception as e:
        print(f"Error saving admin FAISS index: {e}")

# Initialize admin FAISS index
load_admin_faiss_index()

def generate_title_from_message(message: str) -> str:
    words = message.split()
    if len(words) <= 6:
        return message
    return " ".join(words[:6]) + "..."

# --- Text Extraction Functions ---
def extract_text_from_pdf(file_content: bytes) -> str:
    try:
        pdf_file = io.BytesIO(file_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error extracting PDF text: {str(e)}")

def extract_text_from_docx(file_content: bytes) -> str:
    try:
        doc_file = io.BytesIO(file_content)
        doc = docx.Document(doc_file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text.strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error extracting DOCX text: {str(e)}")

def extract_text_from_txt(file_content: bytes) -> str:
    try:
        return file_content.decode('utf-8').strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading text file: {str(e)}")

# --- Helper function to get folder path ---
async def get_folder_path(folder_id: str) -> str:
    if not folder_id:
        return ""
    
    path_parts = []
    current_folder_id = folder_id
    
    while current_folder_id:
        folder = await folders_collection.find_one({"_id": ObjectId(current_folder_id)})
        if not folder:
            break
        path_parts.append(folder["name"])
        current_folder_id = folder.get("parent_id")
    
    path_parts.reverse()
    return " > ".join(path_parts)

# --- Admin Document Functions (with FAISS) ---
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    words = text.split()
    chunks = []
    if chunk_size <= overlap:
        chunk_size = overlap + 1
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk.strip())
    return chunks

async def add_document_to_admin_faiss(text: str, document_name: str, document_id: str, folder_id: str = None):
    global admin_faiss_index, admin_document_texts, admin_document_metadata
    chunks = chunk_text(text)
    if not chunks:
        return

    # Get folder path for metadata
    folder_path = await get_folder_path(folder_id) if folder_id else ""

    embeddings_to_add = []
    metadata_to_add = []

    for i, chunk in enumerate(chunks):
        try:
            embedding = embedding_model.encode([chunk]).astype('float32')
            faiss.normalize_L2(embedding)
            embeddings_to_add.append(embedding[0])

            metadata_to_add.append({
                'document_id': document_id,
                'document_name': document_name,
                'chunk_index': i,
                'text': chunk,
                'type': 'admin',
                'folder_id': folder_id,
                'folder_path': folder_path
            })
        except Exception as e:
            print(f"Error encoding chunk {i} for admin document '{document_name}': {e}")
            continue

    if embeddings_to_add:
        admin_faiss_index.add(np.array(embeddings_to_add))
        admin_document_texts.extend(chunks)
        admin_document_metadata.extend(metadata_to_add)
        save_admin_faiss_index()
        print(f"Added {len(embeddings_to_add)} chunks for admin document '{document_name}' to FAISS.")

def search_admin_documents(query: str, allowed_document_ids: Optional[List[str]] = None, top_k: int = 5) -> List[Dict]:
    if admin_faiss_index.ntotal == 0:
        return []

    query_embedding = embedding_model.encode([query]).astype('float32')
    faiss.normalize_L2(query_embedding)

    # Search for more results to filter and rank
    search_k = min(20, admin_faiss_index.ntotal)
    scores, indices = admin_faiss_index.search(query_embedding, search_k)

    results = []
    seen_documents = set()
    
    for score, idx in zip(scores[0], indices[0]):
        if 0 <= idx < len(admin_document_metadata):
            metadata = admin_document_metadata[idx]
            doc_id = metadata['document_id']
            
            # Filter by allowed documents if specified
            if allowed_document_ids is None or doc_id in allowed_document_ids:
                # Only include one result per document (the best match)
                if doc_id not in seen_documents:
                    result = metadata.copy()
                    result['similarity_score'] = float(score)
                    results.append(result)
                    seen_documents.add(doc_id)
                    
                    if len(results) >= top_k:
                        break
    
    # Sort by similarity score (descending)
    results.sort(key=lambda x: x['similarity_score'], reverse=True)
    return results

# --- Groq API Function ---
async def call_groq_api(messages: List[dict], model: str) -> str:
    if not groq_client:
        raise HTTPException(status_code=500, detail="Groq API key not configured.")

    model_mapping = {
        "llama3-70b-8192": "llama3-70b-8192",
        "gemma2-9b-it": "gemma2-9b-it",
        "meta-llama/llama-4-scout": "meta-llama/llama-4-scout-17b-16e-instruct"
    }

    groq_model_id = model_mapping.get(model)
    if not groq_model_id:
        raise HTTPException(status_code=400, detail=f"Invalid model selected: {model}")

    try:
        completion = groq_client.chat.completions.create(
            model=groq_model_id,
            messages=messages,
            temperature=0.7,
            max_tokens=1024,
            top_p=1,
            stream=False,
            stop=None,
        )
        return completion.choices[0].message.content
    except APIStatusError as e:
        print(f"Groq API error (status {e.status_code}): {e.response}")
        raise HTTPException(status_code=500, detail=f"Groq API error: {e.response.json().get('error', {}).get('message', str(e))}")
    except Exception as e:
        print(f"Unhandled error calling Groq API: {e}")
        raise HTTPException(status_code=500, detail=f"Error calling Groq API: {str(e)}")

# --- Create default admin user ---
async def create_default_admin():
    try:
        admin_exists = await users_collection.find_one({"email": "admin@admin.com"})
        if not admin_exists:
            await users_collection.insert_one({
                "_id": ObjectId(),
                "email": "admin@admin.com",
                "password": hash_password("admin123"),
                "role": "admin",
                "created_at": datetime.utcnow()
            })
            print("Default admin user created: admin@admin.com / admin123")
    except Exception as e:
        print(f"Error creating default admin: {e}")

async def update_existing_sessions_with_user_id():
    """Migration function to add user_id to existing sessions"""
    try:
        admin_user = await users_collection.find_one({"email": "admin@admin.com"})
        if admin_user:
            admin_id = str(admin_user["_id"])
            result = await sessions_collection.update_many(
                {"user_id": {"$exists": False}},
                {"$set": {"user_id": admin_id}}
            )
            print(f"Updated {result.modified_count} existing sessions with admin user_id")
    except Exception as e:
        print(f"Error updating existing sessions: {e}")

@app.on_event("startup")
async def startup_event():
    await create_default_admin()
    await update_existing_sessions_with_user_id()

# --- Authentication endpoints ---
@app.post("/login")
async def login(credentials: LoginRequest):
    email = credentials.username
    password = hash_password(credentials.password)
    
    user = await users_collection.find_one({"email": email, "password": password})
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    access_token = create_access_token(data={"sub": str(user["_id"])})
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": str(user["_id"]),
            "email": user["email"],
            "role": user["role"]
        }
    }

@app.post("/register")
async def register(credentials: LoginRequest):
    email = credentials.username
    password = hash_password(credentials.password)
    
    existing_user = await users_collection.find_one({"email": email})
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    user_data = {
        "_id": ObjectId(),
        "email": email,
        "password": password,
        "role": "user",
        "created_at": datetime.utcnow()
    }
    
    result = await users_collection.insert_one(user_data)
    
    access_token = create_access_token(data={"sub": str(result.inserted_id)})
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": str(result.inserted_id),
            "email": user_data["email"],
            "role": user_data["role"]
        }
    }

@app.get("/me")
async def get_current_user_info(current_user = Depends(get_current_user)):
    return {
        "id": str(current_user["_id"]),
        "email": current_user["email"],
        "role": current_user["role"]
    }

# --- API Endpoints ---
@app.get("/")
async def root():
    return {"message": "Groq Chat Backend API is running."}

# --- NEW: Enhanced Folder-based Document Selection Endpoints ---
@app.get("/folders-with-documents")
async def get_folders_with_documents(folder: str = None, current_user = Depends(get_current_user)):
    """Get folders and documents for a specific folder level"""
    try:
        # Get folders
        if folder and folder != "null":
            folder_filter = {"parent_id": folder}
        else:
            folder_filter = {"parent_id": {"$in": [None, ""]}}
        
        folders_cursor = folders_collection.find(folder_filter)
        folders = await folders_cursor.to_list(length=100)
        
        # Get documents
        if folder and folder != "null":
            doc_filter = {"folder_id": folder}
        else:
            doc_filter = {"folder_id": {"$in": [None, ""]}}
        
        documents_cursor = documents_collection.find(doc_filter, {"text_content": 0})
        documents = await documents_cursor.to_list(length=100)
        
        # Count items in each folder
        for folder_doc in folders:
            folder_id = str(folder_doc["_id"])
            doc_count = await documents_collection.count_documents({"folder_id": folder_id})
            subfolder_count = await folders_collection.count_documents({"parent_id": folder_id})
            folder_doc["item_count"] = doc_count + subfolder_count
        
        return {
            "folders": [
                {
                    "id": str(f["_id"]),
                    "name": f["name"],
                    "parent_id": f.get("parent_id"),
                    "created_date": f["created_date"],
                    "item_count": f.get("item_count", 0)
                } for f in folders
            ],
            "documents": [
                {
                    "id": str(doc["_id"]),
                    "name": doc["name"],
                    "size": doc["size"],
                    "type": doc["type"],
                    "upload_date": doc["upload_date"],
                    "uploaded_by": doc.get("uploaded_by", "admin"),
                    "folder_id": doc.get("folder_id")
                } for doc in documents
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve folders and documents: {str(e)}")

@app.get("/all-folders-documents")
async def get_all_folders_documents(current_user = Depends(get_current_user)):
    """Get all folders and documents for select all functionality"""
    try:
        # Get all folders
        folders_cursor = folders_collection.find({})
        folders = await folders_cursor.to_list(length=1000)
        
        # Get all documents
        documents_cursor = documents_collection.find({}, {"text_content": 0})
        documents = await documents_cursor.to_list(length=1000)
        
        return {
            "folders": [
                {
                    "id": str(f["_id"]),
                    "name": f["name"],
                    "parent_id": f.get("parent_id")
                } for f in folders
            ],
            "documents": [
                {
                    "id": str(doc["_id"]),
                    "name": doc["name"],
                    "folder_id": doc.get("folder_id")
                } for doc in documents
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve all folders and documents: {str(e)}")

@app.get("/folder-documents/{folder_id}")
async def get_folder_documents(folder_id: str, current_user = Depends(get_current_user)):
    """Get all documents within a specific folder (recursive)"""
    try:
        async def get_documents_recursive(fid):
            documents = []
            
            # Get direct documents in folder
            docs_cursor = documents_collection.find({"folder_id": fid}, {"text_content": 0})
            direct_docs = await docs_cursor.to_list(length=100)
            documents.extend(direct_docs)
            
            # Get documents from subfolders
            subfolders_cursor = folders_collection.find({"parent_id": fid})
            subfolders = await subfolders_cursor.to_list(length=100)
            
            for subfolder in subfolders:
                subfolder_docs = await get_documents_recursive(str(subfolder["_id"]))
                documents.extend(subfolder_docs)
            
            return documents
        
        documents = await get_documents_recursive(folder_id)
        
        return {
            "documents": [
                {
                    "id": str(doc["_id"]),
                    "name": doc["name"],
                    "size": doc["size"],
                    "type": doc["type"],
                    "upload_date": doc["upload_date"],
                    "uploaded_by": doc.get("uploaded_by", "admin"),
                    "folder_id": doc.get("folder_id")
                } for doc in documents
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve folder documents: {str(e)}")

# --- Enhanced Chat Endpoint with Folder Support ---
@app.post("/chat")
async def chat(request: ChatRequest, current_user = Depends(get_current_user)):
    try:
        current_time = datetime.utcnow()
        user_id = str(current_user["_id"])
        user_message_for_ai = request.message
        chat_document_ids = []
        references = []

        # Handle document upload in chat (full text processing)
        if request.document_data and request.document_filename:
            try:
                # Decode base64 document
                file_content = base64.b64decode(request.document_data)
                
                # Extract full text based on file type
                filename = request.document_filename.lower()
                if filename.endswith('.pdf'):
                    full_text = extract_text_from_pdf(file_content)
                elif filename.endswith(('.doc', '.docx')):
                    full_text = extract_text_from_docx(file_content)
                elif filename.endswith('.txt'):
                    full_text = extract_text_from_txt(file_content)
                else:
                    raise HTTPException(status_code=400, detail="Unsupported file type")

                if not full_text.strip():
                    raise HTTPException(status_code=400, detail="No text content found in document")

                # Store chat document in database
                chat_doc_data = {
                    "_id": ObjectId(),
                    "name": request.document_filename,
                    "size": len(file_content),
                    "type": "chat_document",
                    "upload_date": current_time,
                    "text_content": full_text,
                    "session_id": request.session_id,
                    "user_id": user_id
                }
                
                chat_doc_result = await chat_documents_collection.insert_one(chat_doc_data)
                chat_document_id = str(chat_doc_result.inserted_id)
                chat_document_ids.append(chat_document_id)

                # Add reference for uploaded document
                references.append({
                    "document_id": chat_document_id,
                    "document_name": request.document_filename,
                    "document_type": "uploaded",
                    "folder_path": None
                })

                # Determine action type
                action_type = request.action_type or 'chat'
                
                if action_type == 'summarize':
                    user_message_for_ai = f"Please provide a comprehensive summary of this document:\n\n{full_text}"
                else:
                    # For chat mode
                    if request.message.strip():
                        # User provided both document and question
                        user_message_for_ai = (
                            f"Based on the following document content, please answer the user's question:\n\n"
                            f"DOCUMENT: {request.document_filename}\n"
                            f"CONTENT:\n{full_text}\n\n"
                            f"USER QUESTION: {request.message}\n\n"
                            f"Please provide a detailed answer based on the document content above."
                        )
                    else:
                        # Only document provided, no specific question
                        user_message_for_ai = (
                            f"I've uploaded a document titled '{request.document_filename}'. "
                            f"Please analyze this document and provide key insights, main points, or ask me what specific information you'd like to know about it.\n\n"
                            f"DOCUMENT CONTENT:\n{full_text}"
                        )

            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Error processing document: {str(e)}")

        # Handle folder-based document selection (ENHANCED)
        else:
            all_selected_document_ids = request.selected_document_ids or []
            
            # If folders are selected, get all documents from those folders
            if request.selected_folder_ids:
                for folder_id in request.selected_folder_ids:
                    try:
                        folder_docs_response = await get_folder_documents(folder_id, current_user)
                        folder_doc_ids = [doc["id"] for doc in folder_docs_response["documents"]]
                        all_selected_document_ids.extend(folder_doc_ids)
                    except Exception as e:
                        print(f"Error getting documents for folder {folder_id}: {e}")
            
            # Remove duplicates
            all_selected_document_ids = list(set(all_selected_document_ids))

            # Use the combined document IDs for RAG
            if all_selected_document_ids and len(all_selected_document_ids) > 0:
                query_for_docs = request.message.strip()
                if admin_faiss_index.ntotal > 0 and query_for_docs:
                    relevant_chunks = search_admin_documents(query_for_docs, allowed_document_ids=all_selected_document_ids, top_k=5)
                    
                    if relevant_chunks:
                        context_parts = []
                        doc_references = {}
                        
                        for chunk in relevant_chunks:
                            folder_info = f" (Folder: {chunk.get('folder_path', 'Root')})" if chunk.get('folder_path') else ""
                            context_parts.append(f"From '{chunk['document_name']}{folder_info}':\n{chunk['text']}")
                            
                            doc_id = chunk['document_id']
                            if doc_id not in doc_references:
                                doc_references[doc_id] = {
                                    "document_id": doc_id,
                                    "document_name": chunk['document_name'],
                                    "document_type": "admin",
                                    "folder_path": chunk.get('folder_path'),
                                    "similarity_score": chunk['similarity_score']
                                }
                        
                        references = list(doc_references.values())
                        combined_context = "\n\n---\n\n".join(context_parts)
                        
                        user_message_for_ai = (
                            f"You are a document assistant. Use the following content from the selected folders and documents to answer the user's question. "
                            f"Provide specific references to the documents and folders you're citing.\n\n"
                            f"CONTEXT:\n{combined_context}\n\n"
                            f"USER QUESTION: {request.message}\n\n"
                            f"Please provide a comprehensive answer based on the above context from the selected folders and documents."
                        )
                    else:
                        ai_response = "No relevant information found in the selected folders and documents for your query."
                        # Store and return response early
                        assistant_message = {
                            "role": "assistant",
                            "content": ai_response,
                            "timestamp": current_time.isoformat(),
                            "references": []
                        }
                        user_message_for_db = {
                            "role": "user",
                            "content": request.message,
                            "timestamp": current_time.isoformat(),
                            "chat_documents": chat_document_ids
                        }
                        if request.image_data:
                            user_message_for_db["image_url"] = request.image_data

                        if not request.session_id:
                            session_data = {
                                "_id": ObjectId(),
                                "user_id": user_id,
                                "title": generate_title_from_message(request.message),
                                "model": request.model,
                                "messages": [user_message_for_db, assistant_message],
                                "created_at": current_time,
                                "updated_at": current_time,
                                "chat_documents": chat_document_ids
                            }
                            result = await sessions_collection.insert_one(session_data)
                            session_id = str(result.inserted_id)
                        else:
                            session = await sessions_collection.find_one({
                                "_id": ObjectId(request.session_id),
                                "user_id": user_id
                            })
                            if session:
                                session["messages"].extend([user_message_for_db, assistant_message])
                                await sessions_collection.update_one(
                                    {"_id": ObjectId(request.session_id)},
                                    {"$set": {"messages": session["messages"], "updated_at": current_time}}
                                )
                            session_id = request.session_id

                        return {"response": ai_response, "session_id": session_id, "references": []}

        # Prepare message for database
        user_message_for_db = {
            "role": "user",
            "content": request.message or f"📄 Uploaded: {request.document_filename}" if request.document_data else request.message,
            "timestamp": current_time.isoformat(),
            "chat_documents": chat_document_ids
        }
        if request.image_data:
            user_message_for_db["image_url"] = request.image_data

        # Handle session creation/updating
        if not request.session_id:
            session_data = {
                "_id": ObjectId(),
                "user_id": user_id,
                "title": generate_title_from_message(request.message or f"Document: {request.document_filename}"),
                "model": request.model,
                "messages": [user_message_for_db],
                "created_at": current_time,
                "updated_at": current_time,
                "chat_documents": chat_document_ids
            }
            result = await sessions_collection.insert_one(session_data)
            session_id = str(result.inserted_id)
            
            # Update chat documents with actual session_id
            if chat_document_ids:
                await chat_documents_collection.update_many(
                    {"_id": {"$in": [ObjectId(doc_id) for doc_id in chat_document_ids]}},
                    {"$set": {"session_id": session_id}}
                )
        else:
            # Verify session belongs to current user
            session = await sessions_collection.find_one({
                "_id": ObjectId(request.session_id),
                "user_id": user_id
            })
            if not session:
                raise HTTPException(status_code=404, detail="Session not found or access denied")

            session["messages"].append(user_message_for_db)
            
            # Update session with new chat documents
            existing_chat_docs = session.get("chat_documents", [])
            existing_chat_docs.extend(chat_document_ids)
            
            await sessions_collection.update_one(
                {"_id": ObjectId(request.session_id)},
                {
                    "$set": {
                        "messages": session["messages"],
                        "updated_at": current_time,
                        "chat_documents": existing_chat_docs
                    }
                }
            )
            session_id = request.session_id

        # Prepare messages for Groq API
        api_messages_for_groq = []
        
        if not request.session_id:
            # New session
            current_api_message_content = []
            current_api_message_content.append({"type": "text", "text": user_message_for_ai})
            if request.image_data and request.model == "meta-llama/llama-4-scout":
                current_api_message_content.append({
                    "type": "image_url", 
                    "image_url": {"url": f"data:image/jpeg;base64,{request.image_data}"}
                })
            api_messages_for_groq.append({"role": "user", "content": current_api_message_content})
        else:
            # Existing session - get recent messages
            session = await sessions_collection.find_one({"_id": ObjectId(session_id)})
            historical_messages = session["messages"][-10:]
            
            for msg in historical_messages:
                if msg["role"] == "user":
                    if msg.get("timestamp") == user_message_for_db["timestamp"]:
                        # Current message
                        content = [{"type": "text", "text": user_message_for_ai}]
                        if request.image_data and request.model == "meta-llama/llama-4-scout":
                            content.append({
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{request.image_data}"}
                            })
                        api_messages_for_groq.append({"role": "user", "content": content})
                    else:
                        # Historical message
                        content = [{"type": "text", "text": msg["content"]}]
                        api_messages_for_groq.append({"role": "user", "content": content})
                else:
                    # Assistant message
                    content = [{"type": "text", "text": msg["content"]}]
                    api_messages_for_groq.append({"role": "assistant", "content": content})

        # Convert to format expected by Groq API
        final_api_messages = []
        for msg in api_messages_for_groq:
            if msg['role'] == 'assistant':
                text_content = " ".join([item.get('text', '') for item in msg['content'] if item.get('type') == 'text'])
                final_api_messages.append({"role": "assistant", "content": text_content.strip()})
            else:
                final_api_messages.append(msg)

        # Call Groq API
        ai_response = await call_groq_api(final_api_messages, request.model)

        # Store assistant response with references
        assistant_message = {
            "role": "assistant",
            "content": ai_response,
            "timestamp": datetime.utcnow().isoformat(),
            "references": references
        }

        await sessions_collection.update_one(
            {"_id": ObjectId(session_id)},
            {
                "$push": {"messages": assistant_message},
                "$set": {"updated_at": datetime.utcnow()}
            }
        )

        return {
            "response": ai_response,
            "session_id": session_id,
            "references": references
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# --- Document Preview Endpoint ---
@app.post("/preview-document")
async def preview_document(request: DocumentPreviewRequest, current_user = Depends(get_current_user)):
    """Get document preview/content for display"""
    try:
        # Try admin documents first
        document = await documents_collection.find_one({"_id": ObjectId(request.document_id)})
        
        if document:
            # Get folder path
            folder_path = await get_folder_path(document.get("folder_id")) if document.get("folder_id") else ""
            
            return {
                "id": str(document["_id"]),
                "name": document["name"],
                "type": document["type"],
                "size": document["size"],
                "upload_date": document["upload_date"],
                "uploaded_by": document.get("uploaded_by", "admin"),
                "folder_path": folder_path,
                "content": document.get("text_content", "")[:2000] + ("..." if len(document.get("text_content", "")) > 2000 else ""),
                "document_type": "admin"
            }
        
        # Try chat documents
        chat_document = await chat_documents_collection.find_one({
            "_id": ObjectId(request.document_id),
            "user_id": str(current_user["_id"])
        })
        
        if chat_document:
            return {
                "id": str(chat_document["_id"]),
                "name": chat_document["name"],
                "type": chat_document["type"],
                "size": chat_document["size"],
                "upload_date": chat_document["upload_date"],
                "uploaded_by": current_user["email"],
                "folder_path": None,
                "content": chat_document.get("text_content", "")[:2000] + ("..." if len(chat_document.get("text_content", "")) > 2000 else ""),
                "document_type": "chat"
            }
        
        raise HTTPException(status_code=404, detail="Document not found")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to preview document: {str(e)}")

# --- Folder Management Endpoints ---
@app.post("/create-folder")
async def create_folder(folder: FolderCreate, current_user = Depends(get_current_user)):
    """Create a new folder (admin only)"""
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Only administrators can create folders")
    
    try:
        # Validate parent folder exists if specified
        if folder.parent_id:
            parent_folder = await folders_collection.find_one({"_id": ObjectId(folder.parent_id)})
            if not parent_folder:
                raise HTTPException(status_code=404, detail="Parent folder not found")
        
        # Check for duplicate folder names in the same parent
        existing_folder = await folders_collection.find_one({
            "name": folder.name,
            "parent_id": folder.parent_id
        })
        if existing_folder:
            raise HTTPException(status_code=400, detail="A folder with this name already exists in the same location")
        
        folder_data = {
            "_id": ObjectId(),
            "name": folder.name,
            "parent_id": folder.parent_id,
            "created_date": datetime.utcnow(),
            "created_by": current_user["email"],
            "item_count": 0
        }
        result = await folders_collection.insert_one(folder_data)
        return {
            "message": "Folder created successfully",
            "folder_id": str(result.inserted_id)
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create folder: {str(e)}")

@app.delete("/folders/{folder_id}")
async def delete_folder(folder_id: str, current_user = Depends(get_current_user)):
    """Delete folder and all contents (admin only)"""
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Only administrators can delete folders")
    
    try:
        async def delete_folder_recursive(fid):
            # Delete all documents in folder and remove from FAISS
            docs_to_delete = await documents_collection.find({"folder_id": fid}).to_list(100)
            for doc in docs_to_delete:
                doc_id = str(doc["_id"])
                # Remove from FAISS index
                global admin_faiss_index, admin_document_texts, admin_document_metadata
                new_document_texts = []
                new_document_metadata = []
                for i, metadata in enumerate(admin_document_metadata):
                    if metadata['document_id'] != doc_id:
                        new_document_texts.append(admin_document_texts[i])
                        new_document_metadata.append(metadata)
                
                admin_faiss_index = faiss.IndexFlatIP(384)
                admin_document_texts = new_document_texts
                admin_document_metadata = new_document_metadata
                
                if admin_document_texts:
                    embeddings = embedding_model.encode(admin_document_texts).astype('float32')
                    faiss.normalize_L2(embeddings)
                    admin_faiss_index.add(embeddings)
                
                save_admin_faiss_index()
            
            await documents_collection.delete_many({"folder_id": fid})
            
            # Get and delete all subfolders recursively
            subfolders = await folders_collection.find({"parent_id": fid}).to_list(100)
            for subfolder in subfolders:
                await delete_folder_recursive(str(subfolder["_id"]))
            
            # Delete the folder itself
            await folders_collection.delete_one({"_id": ObjectId(fid)})
        
        # Check if folder exists
        folder = await folders_collection.find_one({"_id": ObjectId(folder_id)})
        if not folder:
            raise HTTPException(status_code=404, detail="Folder not found")
        
        await delete_folder_recursive(folder_id)
        
        return {"message": "Folder and all contents deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete folder: {str(e)}")

# --- Document Management Endpoints ---
@app.post("/upload-document")
async def upload_document(
    file: UploadFile = File(...), 
    current_user = Depends(get_current_user),
    folder_id: Optional[str] = Form(None)  # <-- THIS IS THE FIX
):
    print(f"Received folder_id: {folder_id}")  # for debugging
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Only administrators can upload documents")
    
    try:
        allowed_types = [
            'application/pdf', 
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 
            'text/plain'
        ]
        if file.content_type not in allowed_types:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}")

        # Validate folder exists if specified
        if folder_id:
            folder = await folders_collection.find_one({"_id": ObjectId(folder_id)})
            if not folder:
                raise HTTPException(status_code=404, detail="Folder not found")

        file_content = await file.read()

        if file.content_type == 'application/pdf':
            text = extract_text_from_pdf(file_content)
        elif file.content_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
            text = extract_text_from_docx(file_content)
        else:
            text = extract_text_from_txt(file_content)

        if not text.strip():
            raise HTTPException(status_code=400, detail="No text content found in document")

        document_data = {
            "_id": ObjectId(),
            "name": file.filename,
            "size": len(file_content),
            "type": file.content_type,
            "upload_date": datetime.utcnow(),
            "uploaded_by": current_user["email"],
            "user_id": str(current_user["_id"]),
            "text_content": text,
            "folder_id": folder_id
        }
        result = await documents_collection.insert_one(document_data)
        document_id = str(result.inserted_id)

        await add_document_to_admin_faiss(text, file.filename, document_id, folder_id)

        return {
            "message": "Admin document uploaded successfully.",
            "document_id": document_id,
            "filename": file.filename,
            "text_length": len(text)
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Error in upload-document endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload document: {str(e)}")
    
@app.get("/documents")
async def get_documents_with_folders(folder: str = None, current_user = Depends(get_current_user)):
    """Get documents and folders"""
    try:
        # Get folders
        if folder and folder != "null":
            folder_filter = {"parent_id": folder}
        else:
            folder_filter = {"parent_id": {"$in": [None, ""]}}
        
        folders_cursor = folders_collection.find(folder_filter)
        folders = await folders_cursor.to_list(length=100)
        
        # Get documents
        if folder and folder != "null":
            doc_filter = {"folder_id": folder}
        else:
            doc_filter = {"folder_id": {"$in": [None, ""]}}
        
        documents_cursor = documents_collection.find(doc_filter, {"text_content": 0})
        documents = await documents_cursor.to_list(length=100)
        
        # Count items in each folder and add folder paths
        for folder_doc in folders:
            folder_id = str(folder_doc["_id"])
            doc_count = await documents_collection.count_documents({"folder_id": folder_id})
            subfolder_count = await folders_collection.count_documents({"parent_id": folder_id})
            folder_doc["item_count"] = doc_count + subfolder_count
            folder_doc["path"] = await get_folder_path(folder_id)
        
        # Add folder paths to documents
        for doc in documents:
            if doc.get("folder_id"):
                doc["folder_path"] = await get_folder_path(doc["folder_id"])
            else:
                doc["folder_path"] = ""
        
        return {
            "folders": [
                {
                    "id": str(f["_id"]),
                    "name": f["name"],
                    "parent_id": f.get("parent_id"),
                    "created_date": f["created_date"],
                    "item_count": f.get("item_count", 0),
                    "path": f.get("path", "")
                } for f in folders
            ],
            "documents": [
                {
                    "id": str(doc["_id"]),
                    "name": doc["name"],
                    "size": doc["size"],
                    "type": doc["type"],
                    "upload_date": doc["upload_date"],
                    "uploaded_by": doc.get("uploaded_by", "admin"),
                    "folder_id": doc.get("folder_id"),
                    "folder_path": doc.get("folder_path", "")
                } for doc in documents
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve documents: {str(e)}")

@app.delete("/documents/{document_id}")
async def delete_document(document_id: str, current_user = Depends(get_current_user)):
    """Delete admin document (admin only)"""
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Only administrators can delete documents")
    
    try:
        document_to_delete = await documents_collection.find_one({"_id": ObjectId(document_id)})
        if not document_to_delete:
            raise HTTPException(status_code=404, detail="Document not found")

        result = await documents_collection.delete_one({"_id": ObjectId(document_id)})
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Document not found")

        # Remove from admin FAISS index
        global admin_faiss_index, admin_document_texts, admin_document_metadata

        new_document_texts = []
        new_document_metadata = []

        for i, metadata in enumerate(admin_document_metadata):
            if metadata['document_id'] != document_id:
                new_document_texts.append(admin_document_texts[i])
                new_document_metadata.append(metadata)

        admin_faiss_index = faiss.IndexFlatIP(384)
        admin_document_texts = new_document_texts
        admin_document_metadata = new_document_metadata

        if admin_document_texts:
            embeddings = embedding_model.encode(admin_document_texts).astype('float32')
            faiss.normalize_L2(embeddings)
            admin_faiss_index.add(embeddings)
        
        save_admin_faiss_index()

        return {"message": "Admin document deleted successfully."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")

# --- Session Management Endpoints ---
@app.get("/sessions", response_model=List[SessionSummary])
async def get_sessions(current_user = Depends(get_current_user)):
    try:
        user_id = str(current_user["_id"])
        sessions = await sessions_collection.find({"user_id": user_id}).sort("updated_at", -1).to_list(100)
        session_summaries = []
        for session in sessions:
            message_count = len(session.get("messages", []))
            session_summaries.append(SessionSummary(
                id=str(session["_id"]),
                title=session["title"],
                message_count=message_count
            ))
        return session_summaries
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve sessions: {str(e)}")

@app.get("/sessions/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str, current_user = Depends(get_current_user)):
    try:
        user_id = str(current_user["_id"])
        session = await sessions_collection.find_one({
            "_id": ObjectId(session_id),
            "user_id": user_id
        })
        if not session:
            raise HTTPException(status_code=404, detail="Session not found or access denied")

        messages = []
        for msg in session.get("messages", []):
            msg_content = ""
            if isinstance(msg.get("content"), list):
                for item in msg["content"]:
                    if item.get("type") == "text":
                        msg_content += item.get("text", "") + " "
            elif isinstance(msg.get("content"), str):
                msg_content = msg["content"]

            messages.append(Message(
                role=msg["role"],
                content=msg_content.strip(),
                image_url=msg.get("image_url"),
                timestamp=msg.get("timestamp"),
                chat_documents=msg.get("chat_documents", []),
                references=msg.get("references", [])
            ))

        return SessionResponse(
            id=str(session["_id"]),
            title=session["title"],
            model=session["model"],
            messages=messages
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve session: {str(e)}")

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str, current_user = Depends(get_current_user)):
    try:
        user_id = str(current_user["_id"])
        session = await sessions_collection.find_one({
            "_id": ObjectId(session_id),
            "user_id": user_id
        })
        if not session:
            raise HTTPException(status_code=404, detail="Session not found or access denied")

        # Delete associated chat documents
        chat_document_ids = session.get("chat_documents", [])
        if chat_document_ids:
            await chat_documents_collection.delete_many({
                "_id": {"$in": [ObjectId(doc_id) for doc_id in chat_document_ids]}
            })

        # Delete session
        result = await sessions_collection.delete_one({
            "_id": ObjectId(session_id),
            "user_id": user_id
        })
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Session not found")
            
        return {"message": "Session and associated documents deleted successfully.", "session_id": session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete session: {str(e)}")

@app.get("/health")
async def health_check():
    try:
        await db.command("ping")
        admin_faiss_status = "initialized" if admin_faiss_index else "not_initialized"
        admin_faiss_doc_count = admin_faiss_index.ntotal if admin_faiss_index else 0
        groq_status = "initialized" if groq_client else "not_initialized"

        return {
            "status": "healthy",
            "database": "connected",
            "admin_faiss_index": admin_faiss_status,
            "admin_documents_indexed": admin_faiss_doc_count,
            "groq_client": groq_status
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
