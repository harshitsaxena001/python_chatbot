import os
import nltk
from chatbot_model import EnhancedChatbotAssistant, check_library_status # Make sure your class is in chatbot_model.py

def initialize_and_train():
    """
    A robust script to initialize, train, and save the chatbot model.
    """
    print("🚀 Initializing SRM NCR Campus Chatbot...")

    # --- Step 1: Download NLTK Data ---
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        print("✅ NLTK data is available.")
    except Exception as e:
        print(f"⚠️  Could not download NLTK data: {e}")

    # --- Step 2: Check for Intents File ---
    intents_file = 'intents_augmented.json'
    if not os.path.exists(intents_file):
        print(f"❌ FATAL: Intents file not found at '{intents_file}'!")
        return

    # --- Step 3: Initialize and Prepare Data ---
    chatbot = EnhancedChatbotAssistant(intents_file)
    
    print("\n📖 Parsing intents from JSON...")
    chatbot.parse_intents()

    if not chatbot.documents:
        print("❌ FATAL: No documents were loaded from the intents file. Check file content and regex.")
        return

    print(f"✅ Found {len(chatbot.documents)} patterns and a vocabulary of {len(chatbot.vocabulary)} words.")

    print("\n📊 Preparing training data (X and y matrices)...")
    chatbot.prepare_data()

    if chatbot.X is None or len(chatbot.X) == 0:
        print("❌ FATAL: Failed to prepare training data.")
        return
        
    print(f"✅ Training data prepared. Shape of X: {chatbot.X.shape}, Shape of y: {chatbot.y.shape}")

    # --- Step 4: Train the Model ---
    print("\n🤖 Starting model training...")
    training_success = chatbot.train_model(epochs=100, verbose=True) # Increased epochs for better training
    # return True

    if not training_success:
        print("❌ Model training failed. Please check the logs above.")
        return

    # --- Step 5: Save the Trained Model ---
    print("\n💾 Saving trained model and dimensions...")
    chatbot.save_model('chatbot_model.pth', 'dimensions.json')

    print("\n🎉 Setup complete! The chatbot is ready to use.")

if __name__ == "__main__":
    # Make sure you have corrected the regex in your chatbot_model.py file first!
    initialize_and_train()
