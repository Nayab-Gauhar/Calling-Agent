# AI Calling Agent — Real-Time Voice Agent

A real-time AI-powered voice phone agent that conducts automated phone conversations using streaming STT → LLM → TTS over Twilio Media Streams, with barge-in support.

## Architecture

```
Caller (Phone) ↔ Twilio ↔ FastAPI WebSocket ↔ Deepgram STT → NVIDIA LLM → Sarvam TTS → Twilio → Caller
```

## Features

- **Inbound & Outbound Calls** via Twilio
- **Real-time Streaming Pipeline** — STT → LLM → TTS, all async/WebSocket
- **Barge-in Support** — caller can interrupt the AI mid-sentence
- **Conversation Persistence** in MongoDB (last 10 messages as context)
- **Call Logging** with duration/timestamps
- **Telegram Notifications** — call summaries sent after each call
- **Pluggable Providers** — swap STT/LLM/TTS modules easily

## Tech Stack

| Component | Provider | Module |
|-----------|----------|--------|
| **Web Framework** | FastAPI + Uvicorn | `app.py` |
| **Telephony** | Twilio Media Streams | `app.py` |
| **STT** | Deepgram (WebSocket) | `modules/deepgram_stt.py` |
| **LLM** | NVIDIA API / Llama 3.3-70B | `modules/nvidia_llm.py` |
| **TTS** | Sarvam AI (bulbul:v3) | `modules/sarvam_tts.py` |
| **Database** | MongoDB (motor) | `modules/mongodb.py` |
| **Notifications** | Telegram Bot | `modules/telegram.py` |
| **Tunnel** | ngrok | `app.py` |

**Alternative providers** (not active, swappable):
- LLM: Google Gemini (`modules/gemini_llm.py`)
- TTS: Deepgram Aura (`modules/deepgram_tts.py`), ElevenLabs (`modules/elevenlabs_tts.py`)

## Setup

1. Clone and install:
   ```bash
   git clone https://github.com/your-repo/ai-calling-agent.git
   cd ai-calling-agent
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env` and fill in your credentials:
   ```bash
   cp .env.example .env
   ```

3. Run:
   ```bash
   python app.py
   ```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Health check |
| `GET/POST` | `/inbound` | Twilio webhook for incoming calls |
| `GET` | `/outbound?phone_number=+91XXXXXXXXXX` | Initiate outbound call |
| `WS` | `/media-stream` | Bidirectional audio stream (Twilio) |

## Project Structure

```
ai-calling-agent/
├── app.py                      # FastAPI server + WebSocket handler
├── config.py                   # Environment variable loader
├── system_prompt.txt           # AI agent personality/instructions
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variable template
├── modules/
│   ├── nvidia_llm.py           # LLM (NVIDIA API, Llama 3.3-70B)
│   ├── deepgram_stt.py         # Speech-to-Text (Deepgram)
│   ├── sarvam_tts.py           # Text-to-Speech (Sarvam AI)
│   ├── mongodb.py              # Database operations
│   ├── telegram.py             # Call notification bot
│   ├── gemini_llm.py           # Alt: Google Gemini LLM
│   ├── deepgram_tts.py         # Alt: Deepgram Aura TTS
│   └── elevenlabs_tts.py       # Alt: ElevenLabs TTS
└── static/                     # Static files
```

## License

MIT
