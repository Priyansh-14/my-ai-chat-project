# My AI Chat Project

A minimal **Streamlit**-based multi-page chat application that uses **OpenAI** for AI responses and can handle both text and images (via OCR).

---

## Features

1. **User Login/Sign Up**

   - Simple email/password authentication (placeholder in this version).
   - Keeps track of `logged_in` status in `st.session_state`.

2. **Multi-Page Structure**

   - **Home Page**: Main chatbot interface.
   - **Login/Sign Up**: Basic auth logic.
   - **Available Bots**: Shows different bots (like StreamlitBot, WeatherBot, etc.).
   - **My Profile**: Shows user details and membership info.
   - **Subscriptions**: Allows users to purchase different subscription plans.

3. **Chat With AI**

   - The app provides a text input for user queries.
   - Also supports file uploads (PDF/images).
   - **PyMuPDF** is used for extracting text from PDFs.
   - **Tesseract** OCR is used to extract text from images.

4. **Streamlit Session State**

   - Maintains conversation history (`past` for user inputs, `generated` for AI responses).
   - Stores user’s login status and email.

5. **OpenAI Integration**
   - The `OPENAI_API_KEY` is securely read from a `.env` file.
   - Uses `ChatOpenAI` with a specific system message restricting the domain to Streamlit questions.

---

## Requirements

- Python 3.8+
- [Streamlit](https://docs.streamlit.io/)
- [OpenAI](https://pypi.org/project/openai/)
- [PyMuPDF](https://pypi.org/project/PyMuPDF/)
- [pytesseract](https://pypi.org/project/pytesseract/)
- [PIL (Pillow)](https://pypi.org/project/Pillow/)
- [python-dotenv](https://pypi.org/project/python-dotenv/)

You also need to have [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) installed on your system if you want to extract text from images.

---

## Getting Started

### 1. Clone This Repository

```bash
git clone https://github.com/Priyansh-14/my-ai-chat-project.git
cd my-ai-chat-project
```

### 2. Create and Activate a Virtual Environment

```bash
python -m venv .venv

# On macOS/Linux:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate.bat
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

```bash
#Create a .env file in the project root with your OpenAI API key:

OPENAI_API_KEY="sk-XXXXXXXXXXXXXXXXXXXX"
```

#### Note: Ensure .env is included in your .gitignore so the API key is not committed.

5. Tesseract Installation

   - If you plan to upload images and need OCR, install Tesseract on your machine:
   - macOS:
     ```bash
     brew install tesseract
     ```
   - Ubuntu/Debian:
     ```bash
     sudo apt-get install tesseract-ocr
     ```
   - Windows: Download the installer

6. Run the App

```bash
streamlit run app.py
```

By default, Streamlit will open the application in your default browser at http://localhost:8501.
