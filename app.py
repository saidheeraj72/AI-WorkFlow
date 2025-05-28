import base64
import os
import tempfile
from flask import Flask, render_template, request, jsonify, session
from groq import Groq
from config import get_db
from bson import ObjectId
from datetime import datetime
from PIL import Image
import io
from bs4 import BeautifulSoup
import re

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev")

# Initialize MongoDB
db = get_db()

# Groq client setup
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
SYSTEM_PROMPT = """
You are a helpful AI assistant. Always format your responses using:
- Headings for main sections
- Bullet points for lists
- Numbered lists for sequences
- Code blocks for code snippets
- Clear paragraph separation
- Bold text for important terms
"""

# Available Groq models with image support
GROQ_MODELS = {
    "gemma2-9b-it": {
        "name": "gemma2-9b-it",
        "description": "Google's lightweight model for efficient performance on standard hardware.Best for daily tasks.",
        "supports_images": False
    },
    "meta-llama/llama-4-scout-17b-16e-instruct": {
        "name": "meta-llama/llama-4-scout",
        "description": "Meta's latest large language model with exceptional Image understandig and OCR capabilities.",
        "supports_images": True
    },
    "llama-3.3-70b-versatile": {
        "name": "llama-3.3-70b-versatile",
        "description": "A powerful open-weight model with strong performance across coding and reasoning tasks.",
        "supports_images": False
    }
}
def format_response(response):
    """Structure the response for better presentation"""
    # Convert to HTML for better formatting
    html_response = response
    
    # Structure code blocks
    html_response = re.sub(r'```([\s\S]*?)```', 
                          r'<pre class="code-block"><code>\1</code></pre>', 
                          html_response)
    
    # Structure lists
    html_response = re.sub(r'(\d+\.\s+.*?)(?=\n\d+\.|\Z)', 
                          r'<li>\1</li>', 
                          html_response, flags=re.DOTALL)
    html_response = html_response.replace('<li>', '<ol><li>').replace('</li>', '</li></ol>')
    
    # Structure headings
    for i in range(3, 0, -1):
        html_response = re.sub(r'\n' + ('#' * i) + r'\s+(.*?)\n', 
                              r'\n<h' + str(i) + r'>\1</h' + str(i) + r'>\n', 
                              html_response)
    
    # Add paragraph tags
    html_response = re.sub(r'\n\n+', r'</p><p>', html_response)
    html_response = f'<p>{html_response}</p>'
    
    # Clean up with BeautifulSoup
    soup = BeautifulSoup(html_response, 'html.parser')
    
    # Add Tailwind classes
    for tag in soup.find_all(['h1', 'h2', 'h3']):
        tag['class'] = 'font-bold my-2'
        
    for tag in soup.find_all('p'):
        tag['class'] = 'my-2'
        
    for tag in soup.find_all('ol'):
        tag['class'] = 'list-decimal pl-5 my-2'
        
    for tag in soup.find_all('ul'):
        tag['class'] = 'list-disc pl-5 my-2'
        
    for tag in soup.find_all('pre'):
        tag['class'] = 'bg-gray-800 p-4 rounded-lg overflow-x-auto my-2'
        
    for tag in soup.find_all('code'):
        tag['class'] = 'text-sm'
    
    return str(soup)
@app.route('/')
def home():
    # Get all chat history for the user
    chat_history = list(db.chats.find({"user_id": session.get("user_id", "default_user")}))
    return render_template('home.html', models=GROQ_MODELS, chat_history=chat_history)

@app.route('/chat')
def chat():
    chat_id = request.args.get('chat_id')
    # Get all chat history for the user
    chat_history = list(db.chats.find({"user_id": session.get("user_id", "default_user")}))
    return render_template('chat.html', models=GROQ_MODELS, chat_history=chat_history, initial_chat_id=chat_id)
@app.route('/api/chat', methods=['POST'])
def chat_api():
    data = request.get_json()
    message = data.get('message', '')
    model = data.get('model', 'meta-llama/llama-4-scout-17b-16e-instruct')
    chat_id = data.get('chat_id')
    image_data = data.get('image')  # Get base64 image data
    
    # Initialize user session
    if 'user_id' not in session:
        session['user_id'] = "user_" + str(datetime.now().timestamp())
    
    # Get or create chat document
    if chat_id:
        chat = db.chats.find_one({"_id": ObjectId(chat_id)})
        if not chat:
            return jsonify({"success": False, "error": "Chat not found"}), 404
    else:
        # Create new chat
        chat = {
            "user_id": session['user_id'],
            "title": message[:30] + "..." if message else "Image chat",
            "messages": [],
            "model": model,
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        }
        result = db.chats.insert_one(chat)
        chat = db.chats.find_one({"_id": result.inserted_id})
    
    # Add user message to history
    message_entry = {
        "role": "user", 
        "content": message,
        "timestamp": datetime.now()
    }
    
    # Add image data if present
    if image_data:
        message_entry["image"] = image_data
    
    chat["messages"].append(message_entry)
    
    try:
        # Prepare messages for Groq API
        messages = []
        
        # Process previous messages
        for m in chat["messages"]:
            # For Llama model with image support
            if model == "meta-llama/llama-4-scout-17b-16e-instruct" and m.get("image"):
                # Create a multimodal message
                messages.append({
                    "role": m["role"],
                    "content": [
                        {"type": "text", "text": m["content"]},
                        {"type": "image_url", "image_url": {"url": m["image"]}}
                    ]
                })
            else:
                # Regular text message
                messages.append({
                    "role": m["role"],
                    "content": m["content"]
                })
        
        # Get response from Groq
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=model,
            temperature=0.7,
            max_tokens=1024
        )
        
        response = chat_completion.choices[0].message.content
        
        # Format the response for better structure
        formatted_response = format_response(response)
        
        # Add assistant response to history
        chat["messages"].append({
            "role": "assistant", 
            "content": formatted_response,
            "timestamp": datetime.now()
        })
        
        chat["updated_at"] = datetime.now()
        
        # Update chat document
        db.chats.update_one(
            {"_id": chat["_id"]},
            {
                "$set": {
                    "messages": chat["messages"],
                    "updated_at": chat["updated_at"],
                    "model": model
                }
            }
        )
        
        return jsonify({
            "success": True,
            "response": formatted_response,  # Send formatted response
            "chat_id": str(chat["_id"]),
            "model": model
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500
# Update the order of your routes
@app.route('/api/chat/history', methods=['GET'])
def get_chat_history():
    # Get all chat history for the user
    chat_history = list(db.chats.find({"user_id": session.get("user_id", "default_user")}))
    # Convert ObjectId to string
    for chat in chat_history:
        chat['_id'] = str(chat['_id'])
    return jsonify({"success": True, "chat_history": chat_history})

# This route MUST come AFTER /api/chat/history
@app.route('/api/chat/<chat_id>', methods=['GET'])
def get_chat(chat_id):
    try:
        # Validate chat_id is a proper ObjectId
        if not ObjectId.is_valid(chat_id):
            return jsonify({"success": False, "error": "Invalid chat ID format"}), 400
            
        chat = db.chats.find_one({"_id": ObjectId(chat_id), "user_id": session.get("user_id", "default_user")})
        if chat:
            chat["_id"] = str(chat["_id"])
            return jsonify({"success": True, "chat": chat})
        return jsonify({"success": False, "error": "Chat not found"}), 404
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/chat/delete/<chat_id>', methods=['DELETE'])
def delete_chat(chat_id):
    result = db.chats.delete_one({"_id": ObjectId(chat_id), "user_id": session.get("user_id", "default_user")})
    if result.deleted_count:
        return jsonify({"success": True})
    return jsonify({"success": False, "error": "Chat not found"}), 404

if __name__ == '__main__':
    app.run(debug=True)