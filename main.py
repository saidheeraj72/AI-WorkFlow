from flask import Flask, render_template, request, jsonify, session
import uuid
from datetime import datetime
import json
import os

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this to a secure secret key

# In-memory storage for demo purposes
# In production, use a proper database
chat_history = {}
available_models = {
    'gpt-3.5-turbo': {
        'name': 'GPT-3.5 Turbo',
        'description': 'The fastest and most cost-effective model, great for general tasks, summarization, and quick creative content.',
        'icon': 'bolt',
        'icon_color': 'text-blue-500'
    },
    'gpt-4': {
        'name': 'GPT-4',
        'description': 'Our most capable model, excelling at complex reasoning, advanced coding, and nuanced understanding.',
        'icon': 'auto_awesome',
        'icon_color': 'text-purple-400'
    },
    'claude-2': {
        'name': 'Claude 2',
        'description': 'Ideal for handling long documents, detailed conversations, and tasks requiring large context windows.',
        'icon': 'description',
        'icon_color': 'text-orange-400'
    },
    'llama-2': {
        'name': 'Llama 2',
        'description': 'A powerful open-source model, excellent for research, customization, and applications requiring transparency.',
        'icon': 'hub',
        'icon_color': 'text-green-400'
    },
    'gemini-pro': {
        'name': 'Gemini Pro',
        'description': "Google's latest model, designed for multimodal capabilities and advanced reasoning across text, code, and images.",
        'icon': 'layers',
        'icon_color': 'text-red-400'
    },
    'bard': {
        'name': 'Bard',
        'description': 'A creative and helpful collaborator, ideal for brainstorming, drafting text, and answering questions conversationally.',
        'icon': 'lightbulb',
        'icon_color': 'text-blue-400'
    }
}

def get_user_id():
    """Get or create user ID for session management"""
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
    return session['user_id']

def get_chat_sessions(user_id):
    """Get all chat sessions for a user"""
    if user_id not in chat_history:
        chat_history[user_id] = {}
    return chat_history[user_id]

def create_new_chat_session(user_id, model='gpt-3.5-turbo'):
    """Create a new chat session"""
    session_id = str(uuid.uuid4())
    timestamp = datetime.now()
    
    if user_id not in chat_history:
        chat_history[user_id] = {}
    
    chat_history[user_id][session_id] = {
        'id': session_id,
        'title': f'New Chat - {timestamp.strftime("%H:%M")}',
        'model': model,
        'created_at': timestamp.isoformat(),
        'messages': [
            {
                'type': 'ai',
                'content': f'Hello! How can I help you today? (Using {available_models[model]["name"]})',
                'timestamp': timestamp.isoformat()
            }
        ]
    }
    
    return session_id

def simulate_ai_response(message, model):
    """Simulate AI response based on the model"""
    responses = {
        'gpt-3.5-turbo': f"GPT-3.5 Turbo response to: {message}",
        'gpt-4': f"GPT-4 advanced response to: {message}",
        'claude-2': f"Claude 2 contextual response to: {message}",
        'llama-2': f"Llama 2 open-source response to: {message}",
        'gemini-pro': f"Gemini Pro multimodal response to: {message}",
        'bard': f"Bard creative response to: {message}"
    }
    
    return responses.get(model, f"AI response to: {message}")

@app.route('/')
def home():
    """Home page with model selection and features"""
    user_id = get_user_id()
    sessions = get_chat_sessions(user_id)
    
    # Get recent chat sessions (last 3)
    recent_sessions = list(sessions.values())[-3:] if sessions else []
    
    return render_template('home.html', 
                         models=available_models,
                         recent_sessions=recent_sessions)

@app.route('/chat')
@app.route('/chat/<session_id>')
def chat(session_id=None):
    """Chat page"""
    user_id = get_user_id()
    sessions = get_chat_sessions(user_id)
    
    # If no session_id provided or session doesn't exist, create new one
    if not session_id or session_id not in sessions:
        model = request.args.get('model', 'gpt-3.5-turbo')
        session_id = create_new_chat_session(user_id, model)
    
    current_session = sessions[session_id]
    recent_sessions = list(sessions.values())[-3:] if sessions else []
    
    return render_template('chat.html', 
                         current_session=current_session,
                         recent_sessions=recent_sessions,
                         models=available_models)

@app.route('/api/send_message', methods=['POST'])
def send_message():
    """Handle sending messages"""
    data = request.json
    message = data.get('message', '').strip()
    session_id = data.get('session_id')
    
    if not message or not session_id:
        return jsonify({'error': 'Message and session_id are required'}), 400
    
    user_id = get_user_id()
    sessions = get_chat_sessions(user_id)
    
    if session_id not in sessions:
        return jsonify({'error': 'Session not found'}), 404
    
    current_session = sessions[session_id]
    timestamp = datetime.now().isoformat()
    
    # Add user message
    user_message = {
        'type': 'user',
        'content': message,
        'timestamp': timestamp
    }
    current_session['messages'].append(user_message)
    
    # Generate AI response
    ai_response_content = simulate_ai_response(message, current_session['model'])
    ai_message = {
        'type': 'ai',
        'content': ai_response_content,
        'timestamp': datetime.now().isoformat()
    }
    current_session['messages'].append(ai_message)
    
    # Update session title if it's the first user message
    if len([msg for msg in current_session['messages'] if msg['type'] == 'user']) == 1:
        current_session['title'] = message[:50] + ('...' if len(message) > 50 else '')
    
    return jsonify({
        'user_message': user_message,
        'ai_message': ai_message,
        'session_title': current_session['title']
    })

@app.route('/api/new_chat', methods=['POST'])
def new_chat():
    """Create a new chat session"""
    data = request.json or {}
    model = data.get('model', 'gpt-3.5-turbo')
    
    user_id = get_user_id()
    session_id = create_new_chat_session(user_id, model)
    
    return jsonify({'session_id': session_id})

@app.route('/api/delete_session/<session_id>', methods=['DELETE'])
def delete_session(session_id):
    """Delete a chat session"""
    user_id = get_user_id()
    sessions = get_chat_sessions(user_id)
    
    if session_id in sessions:
        del sessions[session_id]
        return jsonify({'success': True})
    
    return jsonify({'error': 'Session not found'}), 404

@app.route('/api/change_model', methods=['POST'])
def change_model():
    """Change the model for current session"""
    data = request.json
    session_id = data.get('session_id')
    new_model = data.get('model')
    
    if not session_id or not new_model:
        return jsonify({'error': 'session_id and model are required'}), 400
    
    user_id = get_user_id()
    sessions = get_chat_sessions(user_id)
    
    if session_id not in sessions:
        return jsonify({'error': 'Session not found'}), 404
    
    if new_model not in available_models:
        return jsonify({'error': 'Invalid model'}), 400
    
    sessions[session_id]['model'] = new_model
    
    # Add system message about model change
    timestamp = datetime.now().isoformat()
    system_message = {
        'type': 'ai',
        'content': f'Switched to {available_models[new_model]["name"]}. How can I help you?',
        'timestamp': timestamp
    }
    sessions[session_id]['messages'].append(system_message)
    
    return jsonify({'success': True, 'message': system_message})

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    
    app.run(debug=True)