# My AI Chat Project

A minimal Streamlit-based chat system that accepts text and images (OCR) and uses OpenAI for AI responses.

## 1. Clone the Repository

```bash
git clone https://github.com/<YourUsername>/my-ai-chat-project.git
cd my-ai-chat-project
```

2. Set Up a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate       # On macOS/Linux
# or
.venv\Scripts\activate.bat      # On Windows
```

3. Install Requirements

```bash
pip install -r requirements.txt
```

4. Configure Environment Variables

Create a .env file in the project root:

```bash
OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxx"
```

(Ensure .env is added to .gitignore so your key is never committed.)

5. Run the App

```bash
streamlit run app.py
```

The app will start and you can open it in your web browser (usually at http://localhost:8501).

Enjoy chatting with AI!
