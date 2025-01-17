import os
from dotenv import load_dotenv

load_dotenv()  # Loads variables from .env into environment

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is not set in environment variables.")