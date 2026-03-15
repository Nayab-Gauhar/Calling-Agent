# AI Calling Agent — Real-Time Voice Agent

A real-time AI-powered voice phone agent built with Python. It conducts automated, human-like phone conversations by piping audio through a low-latency streaming pipeline:

```
Caller (Phone) <-> Twilio <-> FastAPI WebSocket <-> Deepgram STT -> Groq LLM -> Sarvam TTS -> Caller
```

## Features

- **Inbound & Outbound Calls** via Twilio Media Streams
- **Real-Time Streaming Pipeline** — STT, LLM, and TTS all operate over async streams for minimal latency
- **Barge-In Support** — caller can interrupt the AI mid-sentence; the system cancels generation and processes new input immediately
- **Conversation Persistence** — chat history stored in MongoDB, last 10 messages loaded as LLM context
- **Call Logging** — each call's metadata (caller, direction, duration) saved to MongoDB
- **Pluggable Providers** — swappable STT/LLM/TTS modules (see below)
- **Bilingual Hindi/English** — configured for Devanagari Hindi with English loan words via Sarvam AI TTS

## Active Stack

| Component | Provider | Model |
|-----------|----------|-------|
| **STT** | Deepgram | WebSocket streaming |
| **LLM** | Groq | Llama 3.1 8B Instant |
| **TTS** | Sarvam AI | Bulbul v3 (Hindi) |

### Alternative Providers (swap in `app.py`)

| Component | Alternative | Module |
|-----------|-------------|--------|
| **LLM** | Google Gemini | `modules/gemini_llm.py` |
| **TTS** | Deepgram Aura | `modules/deepgram_tts.py` |
| **TTS** | ElevenLabs | `modules/elevenlabs_tts.py` |

## Project Structure

```
ai-calling-agent/
├── app.py                      # FastAPI entry point, routes, WebSocket handler
├── config.py                   # Environment variable loader
├── system_prompt.txt           # LLM system prompt
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variable template
├── modules/
│   ├── deepgram_stt.py         # Speech-to-Text (Deepgram WebSocket)
│   ├── groq_llm.py             # LLM (Groq — active)
│   ├── gemini_llm.py           # LLM (Google Gemini — alternative)
│   ├── sarvam_tts.py           # TTS (Sarvam AI — active)
│   ├── deepgram_tts.py         # TTS (Deepgram Aura — alternative)
│   ├── elevenlabs_tts.py       # TTS (ElevenLabs — alternative)
│   └── mongodb.py              # Async MongoDB operations
└── static/                     # Static files directory
```

## Getting Started

### Prerequisites

- Python 3.10+
- A Twilio account with a phone number
- API keys for Deepgram, Groq, and Sarvam AI
- MongoDB instance (optional — app works without it)
- ngrok (for local development)

### Installation

```bash
git clone https://github.com/revolutionarybukhari/ai-calling-agent.git
cd ai-calling-agent
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Configuration

```bash
cp .env.example .env
# Edit .env with your API keys
```

### Running

```bash
python app.py
```

The server starts on `http://localhost:5000`. If `NGROK_AUTH_TOKEN` is set, an ngrok tunnel is automatically created.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Health check / status |
| `GET`, `POST` | `/inbound` | Twilio webhook for incoming calls (returns TwiML) |
| `GET` | `/outbound?phone_number=+1234567890` | Initiate an outbound call |
| `WS` | `/media-stream` | Bidirectional audio stream (used by Twilio) |

## Author

**Syed Husnain Haider Bukhari** — [LinkedIn](https://www.linkedin.com/in/syed-husnain-haider-bukhari/) | [Instagram](https://www.instagram.com/revolutionarybukhari/)

## License

MIT License — see [LICENSE.md](LICENSE.md) for details.
