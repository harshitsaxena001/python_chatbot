

***

# 🤖 AI Assistant Pro — Smart Chatbot with Real‑Time Learning

(static/images/chatbot_s Pro** is a professional‑grade, production‑ready AI chatbot built with **Python (Flask)** and **PyTorch**, featuring:
- **Real‑Time Learning**
- **Session‑based Memory**
- **Advanced Math Calculations**
- **Modern, Responsive Web UI**
- **Feedback & Self‑Improvement System**

Perfect for **enterprise support desks**, **academic institutions**, or **personal knowledge assistants**.  
Ready for **deployment on cloud platforms** like Render, AWS, Azure, Heroku.

***

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🧠 **Real‑Time Learning** | Learns from user feedback instantly — can update `intents_augmented.json` without retraining. |
| 💾 **Session Memory** | Remembers corrections and improvements during a session. |
| 🔢 **Math & Calculations** | Handles basic arithmetic, percentages, powers, roots, and more. |
| 🗂 **Context Awareness** | Keeps track of conversation context for more natural multi‑turn replies. |
| 🖥 **Modern UI** | Clean, responsive design with smooth animations and dark/light mode-ready styling. |
| 📊 **Analytics Dashboard** | View statistics like learned responses, session history, and usage data. |
| 📥 **Export Conversations** | Download chat transcripts for logging or analysis. |
| 📩 **Feedback System** | Correct wrong answers or assign responses to new categories interactively. |

***

## 📂 Project Structure

```
/project-root
├── app.py                   # Flask web server
├── chatbot_model.py          # AI logic & learning system
├── intents_augmented.json    # Training data
├── requirements.txt          # Python dependencies
├── templates/
│   ├── base.html             # Layout template
│   └── index.html            # Chat UI
├── static/
│   ├── css/style.css         # Modern styling
│   ├── js/chat.js            # Frontend logic
│   └── images/               # (Optional) screenshot assets
└── README.md                 # This documentation
```
***
## ✅ Problem Statement
How can we build a **flexible, real-time learning chatbot** to help freshers find basic details about college specially , teachers that:
- Find basic detais about teachers , details about library and canteen etc.
- Engages users through a modern, responsive UI  
- Understands intent and executes math/logic queries  
- Adapts to feedback dynamically (learns from corrections)  
- Retains short-term conversational context  
***
***
## 🚦 Current Progress Status
- **Core Features**: ✅ Completed
  - Intent detection via `intents_augmented.json`
  - Math/logic query execution
  - Flask-based frontend + JS UI
  - Session memory + feedback loop
  - Retraining logic on corrections
- **UI**: ✅ Basic chat interface working
- **Deployment**: 🔄 In progress  
  - Production server setup needed  
  - Persistent JSON storage for adaptive learning
 
***
***
## Pending Work
- Production deployment configuration  
- Secure env variables  
- Analytics dashboard (optional)  
- UI/UX polish for accessibility  

***
***
## 🛠 How the Prototype Solves the Problem
1. **User Input** → User submits a message in the chat UI  
2. **Intent Detection** → Bot checks `intents_augmented.json` and parses math/logic rules  
3. **Response Generation** → Matches intent or executes logic → sends reply  
4. **Feedback Handling** → If user corrects, chatbot:
   - Stores fix in *session memory*
   - Updates `intents_augmented.json`
   - Retrains model on the fly  
5. **Adaptive Learning** → Chatbot improves continuously, no restart needed  
***

## 🛠 Technologies Used

- **Backend**: [Flask](https://flask.palletsprojects.com/)
- **AI/ML**: [PyTorch](https://pytorch.org/), [NLTK](https://www.nltk.org/)
- **Frontend**: HTML5, CSS3, JavaScript (ES6+, Fetch API)
- **Data Storage**: JSON‑based intent learning system
- **Additional**: Responsive CSS variables, Font Awesome icons

***

## 📊 Learning & Feedback Flow
1. **User sends a query**
2. Bot:
   - Matches an intent  
   - Runs math/calculation rules if detected  
   - Falls back to stored learned responses
3. **User gives feedback** (👍 Helpful / ✏ Correct Answer)
4. Bot:
   - Updates **session memory** immediately
   - Optionally **adds/updates `intents_augmented.json`**
   - Retrains in background if necessary

***



### Deployment Checklist:
- Use a production WSGI server (e.g., **gunicorn**)
- Set environment variables for secret keys
- Commit `requirements.txt` with all dependencies
- Maintain `intents_augmented.json` in persistent storage

***


***

## 📜 License

This project is licensed under the **MIT License** — free to use, modify, and distribute.

***

## 📌 Author

**Harshit Saxena** — Developer | AI Enthusiast 
📧 *harshitsaxena0018@gmail.com*  
🐙 [GitHub](https://github.com/harshitsaxena001) | 💼 [LinkedIn](www.linkedin.com/in/harshit-saxena-195130317)

**Vansh Tyagi** - Developer
📧 *vanshtyagi.0107@gmail.com*  
🐙 [GitHub](https://github.com/vansh619-beep) | 💼 [LinkedIn]((www.linkedin.com/in/vansh-tyagi-360057323)

***



***

