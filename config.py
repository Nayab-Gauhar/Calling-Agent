import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Twilio
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER", "")

# Groq LLM
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# NVIDIA NIM LLM
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")

# Google Gemini LLM (alternative)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# Deepgram
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
DEEPGRAM_ENDPOINTING_MS = int(os.getenv("DEEPGRAM_ENDPOINTING_MS", "200"))
DEEPGRAM_LANGUAGE = os.getenv("DEEPGRAM_LANGUAGE", "hi")

# ElevenLabs TTS (alternative)
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "EXAVITQu4vr4xnSDxMaL")

# Sarvam AI TTS
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")

# Telegram Bot (call notifications)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# MongoDB
MONGODB_URI = os.getenv("MONGODB_URI", "")

# Server
NGROK_URL = os.getenv("NGROK_URL", "")
NGROK_AUTH_TOKEN = os.getenv("NGROK_AUTH_TOKEN", "")
PORT = int(os.getenv("PORT", "5000"))

# LLM Settings
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "200"))

_prompt_file = Path(__file__).parent / "system_prompt.txt"
if _prompt_file.exists():
    SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", _prompt_file.read_text().strip())
else:
    SYSTEM_PROMPT = os.getenv(
        "SYSTEM_PROMPT",
        "You are a helpful, friendly, and human-like AI assistant.",
    )
