from flask import Flask, render_template, request, jsonify, session
import os
import json
from datetime import datetime
import secrets
import threading
from collections import defaultdict

try:
    from chatbot_model import EnhancedChatbotAssistant, check_library_status
    CHATBOT_AVAILABLE = True
except Exception as e:
    print(f"Warning: Chatbot model not available: {e}")
    CHATBOT_AVAILABLE = False

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# Configuration
INTENTS_FILE = 'intents_augmented.json'
MODEL_FILE = 'chatbot_model.pth'
DIMS_FILE = 'dimensions.json'
CHAT_HISTORY_FILE = 'chat_history.json'
SESSION_CORRECTIONS_FILE = 'session_corrections.json'

# Session-based memory storage
session_corrections = defaultdict(list)
session_conversations = defaultdict(list)

# Initialize chatbot
if CHATBOT_AVAILABLE:
    function_map = {'library_hours': check_library_status}
    chatbot = EnhancedChatbotAssistant(INTENTS_FILE, function_mappings=function_map)
else:
    chatbot = None

def initialize_chatbot():
    """Initialize the chatbot with error handling"""
    if not CHATBOT_AVAILABLE:
        print("⚠️  Running in demo mode")
        return True
    
    try:
        chatbot.parse_intents()
        
        # Handle model loading
        model_loaded = False
        if os.path.exists(MODEL_FILE) and os.path.exists(DIMS_FILE):
            try:
                chatbot.load_model(MODEL_FILE, DIMS_FILE)
                model_loaded = True
                print("✓ Model loaded successfully!")
            except Exception:
                # Remove incompatible model files
                for file in [MODEL_FILE, DIMS_FILE]:
                    if os.path.exists(file):
                        os.remove(file)
                model_loaded = False
        
        if not model_loaded:
            print("🔄 Training new model...")
            chatbot.prepare_data()
            if chatbot.X is not None and len(chatbot.X) > 0:
                chatbot.train_model(epochs=50)
                chatbot.save_model(MODEL_FILE, DIMS_FILE)
                print("✓ New model trained!")
            else:
                print("❌ No training data available")
                return False
        
        if hasattr(chatbot, 'load_learning_history'):
            chatbot.load_learning_history()
        
        load_session_corrections()
        return True
    except Exception as e:
        print(f"❌ Chatbot initialization error: {e}")
        return False

def save_session_corrections():
    """Save session corrections to file"""
    try:
        data = {session_id: corrections for session_id, corrections in session_corrections.items()}
        with open(SESSION_CORRECTIONS_FILE, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Error saving session corrections: {e}")

def load_session_corrections():
    """Load session corrections from file"""
    try:
        if os.path.exists(SESSION_CORRECTIONS_FILE):
            with open(SESSION_CORRECTIONS_FILE, 'r') as f:
                data = json.load(f)
                for session_id, corrections in data.items():
                    session_corrections[session_id] = corrections
    except Exception as e:
        print(f"Error loading session corrections: {e}")

@app.route('/')
def index():
    """Main chat interface"""
    if 'session_id' not in session:
        session['session_id'] = secrets.token_hex(8)
        session_conversations[session['session_id']] = []
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Enhanced chat endpoint with session memory"""
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        session_id = session.get('session_id')
        
        if not user_message:
            return jsonify({'error': 'Empty message'}), 400
        
        # Store conversation in session
        session_conversations[session_id].append({
            'type': 'user',
            'message': user_message,
            'timestamp': datetime.now().isoformat()
        })
        
        if CHATBOT_AVAILABLE and chatbot:
            # Check session corrections first
            corrected_response = check_session_corrections(session_id, user_message)
            if corrected_response:
                response, intent = corrected_response, "session_corrected"
            else:
                # Check learning history
                historical_response = None
                if hasattr(chatbot, 'check_learning_history'):
                    historical_response = chatbot.check_learning_history(user_message)
                
                if historical_response:
                    response, intent = historical_response, "learned_from_history"
                else:
                    response, intent = chatbot.process_message(user_message)
        else:
            response = f"Demo: You said '{user_message}'"
            intent = "demo_mode"
        
        # Store bot response in session
        session_conversations[session_id].append({
            'type': 'bot',
            'message': response,
            'intent': intent,
            'timestamp': datetime.now().isoformat()
        })
        
        return jsonify({
            'response': response,
            'intent': intent,
            'timestamp': datetime.now().strftime('%H:%M'),
            'learned_from_history': intent == "learned_from_history",
            'session_corrected': intent == "session_corrected"
        })
    
    except Exception as e:
        print(f"Error in chat: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/feedback', methods=['POST'])
def feedback():
    """Enhanced feedback with session memory"""
    try:
        data = request.get_json()
        user_message = data.get('user_message')
        bot_response = data.get('bot_response')
        feedback_type = data.get('feedback')
        correct_response = data.get('correct_response', '').strip()
        session_id = session.get('session_id')
        
        if feedback_type == 'negative' and correct_response:
            # Store correction in session memory
            correction = {
                'user_message': user_message,
                'incorrect_response': bot_response,
                'correct_response': correct_response,
                'timestamp': datetime.now().isoformat()
            }
            
            session_corrections[session_id].append(correction)
            save_session_corrections()
            
            # Also save to permanent learning if chatbot available
            if CHATBOT_AVAILABLE and chatbot and hasattr(chatbot, 'learn_from_manual_correction'):
                chatbot.learn_from_manual_correction(user_message, correct_response)
            
            return jsonify({
                'success': True,
                'message': 'Thank you! I\'ll remember that for this session and learn from it.'
            })
        
        return jsonify({
            'success': True,
            'message': 'Thank you for your feedback!'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def check_session_corrections(session_id, user_message):
    """Check if we have a session-specific correction"""
    if session_id not in session_corrections:
        return None
    
    user_message_lower = user_message.lower().strip()
    
    for correction in session_corrections[session_id]:
        stored_message = correction['user_message'].lower().strip()
        if user_message_lower == stored_message or similarity(user_message_lower, stored_message) > 0.8:
            return correction['correct_response']
    
    return None

def similarity(a, b):
    """Simple similarity check"""
    from difflib import SequenceMatcher
    return SequenceMatcher(None, a, b).ratio()

@app.route('/get_intents')
def get_intents():
    """Get available intents"""
    if CHATBOT_AVAILABLE and chatbot and hasattr(chatbot, 'intents'):
        return jsonify({'intents': chatbot.intents})
    return jsonify({'intents': ['greeting', 'library_hours', 'math_calculation']})

@app.route('/session_stats')
def session_stats():
    """Get session statistics"""
    session_id = session.get('session_id')
    corrections_count = len(session_corrections.get(session_id, []))
    conversations_count = len(session_conversations.get(session_id, []))
    
    return jsonify({
        'session_corrections': corrections_count,
        'session_conversations': conversations_count,
        'recent_corrections': session_corrections.get(session_id, [])[-5:]
    })

@app.route('/clear_session', methods=['POST'])
def clear_session():
    """Clear current session data"""
    session_id = session.get('session_id')
    if session_id:
        session_corrections[session_id] = []
        session_conversations[session_id] = []
        save_session_corrections()
    
    return jsonify({'success': True, 'message': 'Session cleared successfully'})

@app.errorhandler(404)
def not_found(error):
    return render_template('base.html'), 404

if __name__ == '__main__':
    print("🚀 Starting Professional AI Chatbot...")
    
    # Ensure directories exist
    for dir_name in ['templates', 'static', 'static/css', 'static/js']:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
    
    # Create basic intents if missing
    if not os.path.exists(INTENTS_FILE):
        basic_intents = {
            "intents": [
                {
                    "tag": "greeting",
                    "patterns": ["Hi", "Hello", "Hey", "Good morning", "Hi there"],
                    "responses": ["Hello! How can I help you today?", "Hi there! What can I assist you with?"]
                },
                {
                    "tag": "goodbye",
                    "patterns": ["Bye", "Goodbye", "See you later", "Thanks"],
                    "responses": ["Goodbye! Have a great day!", "Thank you for chatting!"]
                }
            ]
        }
        with open(INTENTS_FILE, 'w') as f:
            json.dump(basic_intents, f, indent=2)
    
    initialize_chatbot()
    print("🌐 Server starting at http://127.0.0.1:5000")
    app.run(debug=True, host='127.0.0.1', port=5000, threaded=True)
