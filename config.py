import os
from dotenv import load_dotenv

load_dotenv()

# Twilio
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER", "")

# Groq LLM
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-8b-8192")

# Deepgram
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
DEEPGRAM_ENDPOINTING_MS = int(os.getenv("DEEPGRAM_ENDPOINTING_MS", "200"))

# Sarvam AI TTS
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")

# MongoDB
MONGODB_URI = os.getenv("MONGODB_URI", "")

# Server
NGROK_URL = os.getenv("NGROK_URL", "")
NGROK_AUTH_TOKEN = os.getenv("NGROK_AUTH_TOKEN", "")
PORT = int(os.getenv("PORT", "5000"))

# LLM Settings
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "150"))
with open("system_prompt.txt", "r") as f:
    SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", f.read().strip())
