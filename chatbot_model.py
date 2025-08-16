import os
import json
import random
import datetime
import re
import math
import threading
from difflib import SequenceMatcher
from collections import defaultdict


import nltk
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset



class ChatbotModel(nn.Module):
    """Enhanced neural network model for the chatbot."""
    def __init__(self, input_size, output_size):
        super(ChatbotModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.batch_norm1 = nn.BatchNorm1d(256)
        self.batch_norm2 = nn.BatchNorm1d(128)

    def forward(self, x):
        x = self.relu(self.batch_norm1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.batch_norm2(self.fc2(x)))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x

class EnhancedChatbotAssistant:
    """Enhanced chatbot with real-time learning and dynamic intent updates."""
    
    def __init__(self, intents_path, function_mappings=None):
        self.intents_path = intents_path
        self.model = None
        self.function_mappings = function_mappings
        self.learning_history_file = 'learning_history.json'
        self.learning_history = {}
        self.dynamic_intents_file = 'dynamic_intents.json'
        self.user_corrections = defaultdict(list)
        
        # Thread safety for real-time updates
        self.update_lock = threading.Lock()
        
        # Learning thresholds
        self.confidence_threshold = 0.75
        self.similarity_threshold = 0.85
        self.auto_create_intent_threshold = 3  # Create new intent after 3 similar corrections

        self.documents = []
        self.vocabulary = []
        self.intents = []
        self.intents_responses = {}
        
        self.conversation_context = None
        self.X = None
        self.y = None
        
        # Load or create dynamic intents
        self.load_dynamic_intents()

    def load_dynamic_intents(self):
        """Load dynamically created intents from user feedback."""
        try:
            if os.path.exists(self.dynamic_intents_file):
                with open(self.dynamic_intents_file, 'r', encoding='utf-8') as f:
                    self.dynamic_intents = json.load(f)
                print(f"✓ Loaded {len(self.dynamic_intents)} dynamic intents")
            else:
                self.dynamic_intents = {}
        except Exception as e:
            print(f"Error loading dynamic intents: {e}")
            self.dynamic_intents = {}

    def save_dynamic_intents(self):
        """Save dynamically created intents."""
        try:
            with open(self.dynamic_intents_file, 'w', encoding='utf-8') as f:
                json.dump(self.dynamic_intents, f, indent=2)
        except Exception as e:
            print(f"Error saving dynamic intents: {e}")

    def add_pattern_to_intent_realtime(self, pattern, intent_tag):
        """Add a new pattern to existing intent in real-time."""
        with self.update_lock:
            try:
                # Read current intents file
                with open(self.intents_path, 'r', encoding='utf-8') as f:
                    intents_data = json.load(f)
                
                # Find the intent and add pattern
                intent_found = False
                for intent in intents_data['intents']:
                    if intent['tag'] == intent_tag:
                        if pattern not in intent['patterns']:
                            intent['patterns'].append(pattern)
                            intent_found = True
                            print(f"✓ Added pattern '{pattern}' to intent '{intent_tag}'")
                        break
                
                if intent_found:
                    # Write back to file
                    with open(self.intents_path, 'w', encoding='utf-8') as f:
                        json.dump(intents_data, f, indent=2)
                    
                    # Trigger model retraining in background
                    threading.Thread(target=self.retrain_model_background, daemon=True).start()
                    return True
                else:
                    print(f"Intent '{intent_tag}' not found")
                    return False
                    
            except Exception as e:
                print(f"Error adding pattern to intent: {e}")
                return False

    def create_new_intent_from_corrections(self, patterns, responses, suggested_tag=None):
        """Create a new intent from user corrections."""
        with self.update_lock:
            try:
                # Generate intent tag
                if suggested_tag:
                    intent_tag = suggested_tag.lower().replace(' ', '_')
                else:
                    # Auto-generate tag based on common words in patterns
                    all_words = []
                    for pattern in patterns:
                        words = re.findall(r'\w+', pattern.lower())
                        all_words.extend(words)
                    
                    # Find most common meaningful words
                    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'what', 'how', 'when', 'where', 'why'}
                    meaningful_words = [word for word in set(all_words) if word not in stop_words and len(word) > 2]
                    
                    if meaningful_words:
                        intent_tag = f"custom_{meaningful_words[0]}"
                    else:
                        intent_tag = f"custom_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

                # Read current intents
                with open(self.intents_path, 'r', encoding='utf-8') as f:
                    intents_data = json.load(f)
                
                # Create new intent
                new_intent = {
                    "tag": intent_tag,
                    "patterns": patterns,
                    "responses": responses,
                    "created_by": "user_feedback",
                    "created_at": datetime.datetime.now().isoformat()
                }
                
                # Add to intents
                intents_data['intents'].append(new_intent)
                
                # Save to file
                with open(self.intents_path, 'w', encoding='utf-8') as f:
                    json.dump(intents_data, f, indent=2)
                
                print(f"✓ Created new intent '{intent_tag}' with {len(patterns)} patterns")
                
                # Trigger model retraining
                threading.Thread(target=self.retrain_model_background, daemon=True).start()
                return intent_tag
                
            except Exception as e:
                print(f"Error creating new intent: {e}")
                return None
            


    def retrain_model_background(self):
        """Retrain the model in background thread."""
        try:
            print("🔄 Retraining model with new data...")
            self.parse_intents()
            self.prepare_data()
            if self.X is not None and len(self.X) > 0:
                self.train_model(epochs=30, verbose=False)  # Quick retrain
                self.save_model('chatbot_model.pth', 'dimensions.json')
                print("✓ Model retrained successfully")
            else:
                print("❌ No training data available for retraining")
        except Exception as e:
            print(f"Error retraining model: {e}")

    def learn_from_manual_correction(self, user_message, correct_response, suggested_category=None):
        """Learn from manual user corrections and create new intents if needed."""
        normalized_message = self.normalize_message(user_message)
        
        # Store the correction
        correction_entry = {
            'message': user_message,
            'normalized': normalized_message,
            'response': correct_response,
            'category': suggested_category,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        # Group similar corrections
        similar_key = self.find_similar_correction_group(normalized_message)
        if not similar_key:
            similar_key = normalized_message
        
        self.user_corrections[similar_key].append(correction_entry)
        
        # Save to learning history immediately
        self.save_to_learning_history(user_message, correct_response)
        
        # Check if we should create a new intent
        if len(self.user_corrections[similar_key]) >= self.auto_create_intent_threshold:
            self.suggest_intent_creation(similar_key)
        
        return True

    def find_similar_correction_group(self, normalized_message):
        """Find existing correction group with similar messages."""
        for existing_key in self.user_corrections.keys():
            if self.calculate_similarity(normalized_message, existing_key) > 0.8:
                return existing_key
        return None

    def suggest_intent_creation(self, correction_group_key):
        """Suggest creating a new intent from accumulated corrections."""
        corrections = self.user_corrections[correction_group_key]
        
        if len(corrections) < self.auto_create_intent_threshold:
            return False
        
        # Extract patterns and responses
        patterns = [corr['message'] for corr in corrections]
        responses = list(set([corr['response'] for corr in corrections]))  # Unique responses
        
        # Check if we have a suggested category
        categories = [corr.get('category') for corr in corrections if corr.get('category')]
        suggested_category = categories[0] if categories else None
        
        # Create new intent
        new_intent_tag = self.create_new_intent_from_corrections(
            patterns, responses, suggested_category
        )
        
        if new_intent_tag:
            # Clear the corrections group since we've created an intent
            del self.user_corrections[correction_group_key]
            return new_intent_tag
        
        return False

    def get_correction_suggestions(self):
        """Get suggestions for creating new intents from corrections."""
        suggestions = []
        
        for group_key, corrections in self.user_corrections.items():
            if len(corrections) >= 2:  # Suggest when we have 2+ similar corrections
                patterns = [corr['message'] for corr in corrections]
                responses = list(set([corr['response'] for corr in corrections]))
                
                suggestion = {
                    'group_key': group_key,
                    'pattern_count': len(corrections),
                    'sample_patterns': patterns[:3],  # First 3 patterns as examples
                    'sample_responses': responses[:2],  # First 2 responses
                    'can_auto_create': len(corrections) >= self.auto_create_intent_threshold
                }
                suggestions.append(suggestion)
        
        return suggestions

    def create_intent_from_suggestion(self, group_key, custom_tag=None):
        """Create intent from a specific suggestion."""
        if group_key not in self.user_corrections:
            return False
        
        corrections = self.user_corrections[group_key]
        patterns = [corr['message'] for corr in corrections]
        responses = list(set([corr['response'] for corr in corrections]))
        
        new_intent_tag = self.create_new_intent_from_corrections(
            patterns, responses, custom_tag
        )
        
        if new_intent_tag:
            del self.user_corrections[group_key]
            return new_intent_tag
        
        return False

    @staticmethod
    def tokenize_and_lemmatize(text):
        try:
            lemmatizer = nltk.WordNetLemmatizer()
            words = nltk.word_tokenize(text)
            words = [lemmatizer.lemmatize(word.lower()) for word in words]
            return words
        except Exception as e:
            # Fallback to simple tokenization if NLTK fails
            return text.lower().split()

    def bag_of_words(self, words):
        return np.array([1 if word in words else 0 for word in self.vocabulary], dtype=np.float32)

    def parse_intents(self):
        self.documents = []
        self.vocabulary = []
        self.intents = []
        self.intents_responses = {}

        if os.path.exists(self.intents_path):
            with open(self.intents_path, 'r', encoding='utf-8') as f:
                intents_data = json.load(f)

            for intent in intents_data['intents']:
                tag = intent['tag']
                if tag not in self.intents:
                    self.intents.append(tag)
                    self.intents_responses[tag] = intent['responses']

                for pattern in intent['patterns']:
                    pattern_words = self.tokenize_and_lemmatize(pattern)
                    self.vocabulary.extend(pattern_words)
                    self.documents.append((pattern_words, tag))

            self.vocabulary = sorted(list(set(self.vocabulary)))
        else:
            print(f"Error: Intents file not found at {self.intents_path}")

    def prepare_data(self):
        if not self.documents:
            return

        bags = []
        indices = []

        for document in self.documents:
            words, intent_tag = document
            bag = self.bag_of_words(words)
            intent_index = self.intents.index(intent_tag)
            bags.append(bag)
            indices.append(intent_index)

        self.X = np.array(bags)
        self.y = np.array(indices)

    def train_model(self, batch_size=16, lr=0.001, epochs=100, verbose=True):
        if self.X is None or self.y is None:
            return

        X_tensor = torch.from_numpy(self.X)
        y_tensor = torch.from_numpy(self.y).long()

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model = ChatbotModel(self.X.shape[1], len(self.intents))

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)

        if verbose:
            print(f"\n--- Training Enhanced Model for {epochs} epochs ---")
        
        for epoch in range(epochs):
            running_loss = 0.0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            if verbose and (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs}: Loss: {running_loss / len(loader):.4f}")
        
        if verbose:
            print("--- Enhanced Model Training Complete ---\n")

    def save_model(self, model_path, dimensions_path):
        if self.model:
            torch.save(self.model.state_dict(), model_path)
            with open(dimensions_path, 'w', encoding='utf-8') as f:
                json.dump({'input_size': self.X.shape[1], 'output_size': len(self.intents)}, f)

    def load_model(self, model_path, dimensions_path):
        with open(dimensions_path, 'r', encoding='utf-8') as f:
            dimensions = json.load(f)
        
        self.model = ChatbotModel(dimensions['input_size'], dimensions['output_size'])
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.eval()

    # [Include all other methods from previous implementation...]
    # (load_learning_history, save_to_learning_history, check_learning_history, etc.)
    
    def load_learning_history(self):
        """Load learning history from file"""
        try:
            if os.path.exists(self.learning_history_file):
                with open(self.learning_history_file, 'r', encoding='utf-8') as f:
                    self.learning_history = json.load(f)
                print(f"✓ Loaded {len(self.learning_history)} learning entries from history")
            else:
                self.learning_history = {}
        except Exception as e:
            print(f"Error loading learning history: {e}")
            self.learning_history = {}

    def save_to_learning_history(self, user_message, correct_response):
        """Save user corrections to learning history"""
        try:
            normalized_message = self.normalize_message(user_message)
            
            self.learning_history[normalized_message] = {
                'response': correct_response,
                'original_message': user_message,
                'learned_at': datetime.datetime.now().isoformat(),
                'usage_count': 0
            }
            
            with open(self.learning_history_file, 'w', encoding='utf-8') as f:
                json.dump(self.learning_history, f, indent=2)
            
            print(f"✓ Saved learning entry: {user_message[:50]}...")
        except Exception as e:
            print(f"Error saving to learning history: {e}")

    def check_learning_history(self, user_message):
        """Check if we have a learned response for this message"""
        try:
            normalized_message = self.normalize_message(user_message)
            
            if normalized_message in self.learning_history:
                entry = self.learning_history[normalized_message]
                entry['usage_count'] = entry.get('usage_count', 0) + 1
                return entry['response']
            
            # Fuzzy matching
            best_match = None
            best_similarity = 0.0
            
            for learned_msg, entry in self.learning_history.items():
                similarity = self.calculate_similarity(normalized_message, learned_msg)
                if similarity > self.similarity_threshold and similarity > best_similarity:
                    best_similarity = similarity
                    best_match = entry
            
            if best_match:
                best_match['usage_count'] = best_match.get('usage_count', 0) + 1
                return best_match['response']
            
            return None
        except Exception as e:
            print(f"Error checking learning history: {e}")
            return None

    def normalize_message(self, message):
        """Normalize message for better matching"""
        normalized = message.lower().strip()
        normalized = re.sub(r'[^\w\s]', '', normalized)
        normalized = re.sub(r'\s+', ' ', normalized)
        return normalized

    def calculate_similarity(self, msg1, msg2):
        """Calculate similarity between two messages"""
        return SequenceMatcher(None, msg1, msg2).ratio()

    def get_learning_stats(self):
        """Get comprehensive learning statistics"""
        total_entries = len(self.learning_history)
        total_usage = sum(entry.get('usage_count', 0) for entry in self.learning_history.values())
        
        # Correction suggestions
        suggestions = self.get_correction_suggestions()
        
        # Most used responses
        most_used = sorted(
            self.learning_history.items(), 
            key=lambda x: x[1].get('usage_count', 0), 
            reverse=True
        )[:5]
        
        return {
            'total_learned_responses': total_entries,
            'total_usage_count': total_usage,
            'pending_corrections': len(self.user_corrections),
            'correction_suggestions': suggestions,
            'most_used_responses': [
                {
                    'message': entry[1]['original_message'][:50] + '...' if len(entry[1]['original_message']) > 50 else entry[1]['original_message'],
                    'usage_count': entry[1].get('usage_count', 0)
                }
                for entry in most_used if entry[1].get('usage_count', 0) > 0
            ]
        }

    # [Include all mathematical calculation methods from previous implementation...]
    def calculate_math_expression(self, expression):
        """Safely evaluate mathematical expressions"""
        try:
            expression = re.sub(r'[^0-9+\-*/().\s]', '', expression)
            
            replacements = {
                'plus': '+', 'add': '+', 'added to': '+',
                'minus': '-', 'subtract': '-', 'subtracted from': '-',
                'times': '*', 'multiply': '*', 'multiplied by': '*',
                'divided by': '/', 'divide': '/', 'over': '/',
                'x': '*', 'X': '*'
            }
            
            for word, operator in replacements.items():
                expression = expression.replace(word, operator)
            
            if re.match(r'^[0-9+\-*/().\s]+$', expression.strip()):
                result = eval(expression.strip())
                
                if isinstance(result, float):
                    if result.is_integer():
                        result = int(result)
                    else:
                        result = round(result, 6)
                
                return f"The result is: {result}"
            else:
                return "Please provide a valid mathematical expression using numbers and basic operators (+, -, *, /)."
                
        except ZeroDivisionError:
            return "Error: Division by zero is not allowed."
        except Exception as e:
            return "Sorry, I couldn't calculate that. Please check your expression and try again."

    def advanced_math_calculator(self, user_input):
        """Handle complex math operations"""
        try:
            user_input_lower = user_input.lower()
            
            # Square root calculation
            if 'square root' in user_input_lower:
                numbers = re.findall(r'\d+\.?\d*', user_input)
                if numbers:
                    num = float(numbers[0])
                    if num >= 0:
                        result = math.sqrt(num)
                        return f"The square root of {num} is {result:.4f}"
                    else:
                        return "Cannot calculate square root of negative numbers."
            
            # Percentage calculation
            if 'percent' in user_input_lower or '%' in user_input:
                numbers = re.findall(r'\d+\.?\d*', user_input)
                if len(numbers) >= 2:
                    percentage = float(numbers[0])
                    total = float(numbers[1])
                    result = (percentage / 100) * total
                    return f"{percentage}% of {total} is {result}"
                elif len(numbers) == 1:
                    return f"I found the number {numbers[0]}. Please specify what you want to calculate the percentage of."
            
            # Power calculation
            if 'power' in user_input_lower or '^' in user_input or '**' in user_input:
                power_pattern = r'(\d+\.?\d*)\s*(?:to the power of|\^|\*\*)\s*(\d+\.?\d*)'
                match = re.search(power_pattern, user_input_lower)
                if match:
                    base = float(match.group(1))
                    exponent = float(match.group(2))
                    result = base ** exponent
                    return f"{base} to the power of {exponent} is {result}"
            
            # Factorial calculation
            if 'factorial' in user_input_lower:
                numbers = re.findall(r'\d+', user_input)
                if numbers:
                    num = int(numbers[0])
                    if 0 <= num <= 20:
                        result = math.factorial(num)
                        return f"The factorial of {num} is {result}"
                    else:
                        return "Factorial calculation is limited to numbers between 0 and 20."
            
            # Basic arithmetic expression
            expression_pattern = r'[\d+\-*/().\s]+'
            expression_match = re.search(expression_pattern, user_input)
            if expression_match and any(op in user_input for op in ['+', '-', '*', '/']):
                expression = expression_match.group()
                return self.calculate_math_expression(expression)
            
            return "I couldn't find a mathematical expression to calculate. Please provide a clear math problem like '2 + 3', 'square root of 16', or '20% of 150'."
            
        except Exception as e:
            return f"Sorry, I encountered an error while calculating: {str(e)}"

    def process_message(self, input_message):
        """Enhanced message processing with real-time learning"""
        if not self.model:
            return "The model is not loaded.", None

        # Check for mathematical expressions first
        math_keywords = ['calculate', 'what is', 'solve', '+', '-', '*', '/', '=', 'math', 
                        'square root', 'percentage', '%', 'factorial', 'power', '^']
        
        if any(keyword in input_message.lower() for keyword in math_keywords):
            if (re.search(r'\d+.*[+\-*/].*\d+', input_message) or 
                re.search(r'(square root|percentage|power|factorial|\^|\*\*|%)', input_message.lower()) or
                any(op in input_message for op in ['+', '-', '*', '/'])):
                
                math_result = self.advanced_math_calculator(input_message)
                return math_result, "math_calculation"

        # Check context handling
        if self.conversation_context:
            contextual_response = self.handle_contextual_reply(input_message)
            if contextual_response:
                return contextual_response, "context_resolved"

        # Process with neural network
        words = self.tokenize_and_lemmatize(input_message)
        bag = self.bag_of_words(words)
        bag_tensor = torch.from_numpy(bag).unsqueeze(0)

        with torch.no_grad():
            predictions = self.model(bag_tensor)
        
        probabilities = torch.softmax(predictions, dim=1)
        confidence, predicted_class_index = torch.max(probabilities, dim=1)

        if confidence.item() > self.confidence_threshold:
            predicted_intent = self.intents[predicted_class_index.item()]
            
            if self.function_mappings and predicted_intent in self.function_mappings:
                return self.function_mappings[predicted_intent](), predicted_intent

            return random.choice(self.intents_responses[predicted_intent]), predicted_intent
        else:
            user_words = set(words)
            
            if 'library' in user_words:
                self.conversation_context = 'library_suggestion'
                return "I see you mentioned the library. Are you looking for its hours, services, or directions?", "suggestion"
            if 'exam' in user_words or 'test' in user_words or 'datesheet' in user_words:
                return "It sounds like you're asking about exams. Do you want to know about the schedule?", "suggestion"
            if 'canteen' in user_words or 'food' in user_words or 'lunch' in user_words:
                return "Are you asking about the canteen menu?", "suggestion"
            if 'fee' in user_words or 'payment' in user_words:
                return "It seems you have a query about fees. Are you asking how to pay them?", "suggestion"
            
            return "I'm not sure I understand. Can you rephrase or teach me the correct response?", "fallback"

    def handle_contextual_reply(self, message):
        """Handle contextual replies"""
        if self.conversation_context == 'library_suggestion':
            self.conversation_context = None 
            
            msg_lower = message.lower()
            if 'hour' in msg_lower or 'timing' in msg_lower:
                return random.choice(self.intents_responses['library_hours'])
            if 'service' in msg_lower:
                return random.choice(self.intents_responses['library_services'])
            if 'direction' in msg_lower or 'route' in msg_lower:
                return random.choice(self.intents_responses['campus_navigation'])
        
        return None

    def learn_from_feedback(self, incorrect_message, correct_intent_tag):
        """Enhanced learning from feedback with real-time updates"""
        if correct_intent_tag not in self.intents:
            return False

        return self.add_pattern_to_intent_realtime(incorrect_message, correct_intent_tag)

def check_library_status():
    now = datetime.datetime.now().time()
    if now >= datetime.time(9, 0) and now < datetime.time(20, 0):
        return "The library is currently OPEN. It closes at 8 PM."
    else:
        return "The library is currently CLOSED. It opens at 9 AM."

# Create an alias for backward compatibility
ChatbotAssistant = EnhancedChatbotAssistant
