"""
AI Calling Agent — Real-Time Voice Agent
Uses FastAPI + Twilio Media Streams + Deepgram STT + Gemini LLM + ElevenLabs TTS
for human-like, low-latency phone conversations with barge-in support.
"""

import json
import base64
import asyncio
import time
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse, Connect

from config import (
    TWILIO_ACCOUNT_SID,
    TWILIO_AUTH_TOKEN,
    TWILIO_PHONE_NUMBER,
    NGROK_URL,
    NGROK_AUTH_TOKEN,
    PORT,
    SYSTEM_PROMPT,
)
from modules.deepgram_stt import DeepgramSTT
from modules.groq_llm import GroqLLM
from modules.sarvam_tts import SarvamTTS
from modules.mongodb import (
    get_chat_history,
    save_chat_history,
    append_to_chat_history,
    save_call_log,
)

app = FastAPI(title="AI Calling Agent", version="2.0.0")
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Serve static files (for any pre-recorded audio like greetings)
app.mount("/static", StaticFiles(directory="static"), name="static")

# ─── Dynamic ngrok URL (set at startup if auto-tunnel is used) ───
_ngrok_domain = NGROK_URL  # Will be overridden if ngrok auto-starts


def get_ngrok_url() -> str:
    """Get the current ngrok domain."""
    return _ngrok_domain


# ─── Health Check ─────────────────────────────────────────────────

@app.get("/")
async def index():
    return JSONResponse({
        "message": "AI Calling Agent — Real-Time Voice Agent v2.0",
        "status": "running",
        "endpoints": {
            "inbound": "POST /inbound",
            "outbound": "GET /outbound?phone_number=+1234567890",
            "media_stream": "WS /media-stream",
        },
    })


# ─── Inbound Call Handler ────────────────────────────────────────

@app.api_route("/inbound", methods=["GET", "POST"])
async def inbound_call(request: Request):
    """
    Twilio webhook for incoming calls.
    Returns TwiML that connects the call to our WebSocket media stream.
    """
    response = VoiceResponse()

    # Greet the caller briefly to minimize delay
    response.say(
        "Hi there! How can I help you?",
        voice="Polly.Joanna",
    )
    response.pause(length=1)

    # Connect to bidirectional media stream
    connect = Connect()
    stream = connect.stream(
        url=f"wss://{get_ngrok_url()}/media-stream",
    )
    # Pass caller info as custom parameters
    form_data = await request.form()
    caller_number = form_data.get("From", "unknown")
    stream.parameter(name="caller_number", value=caller_number)
    stream.parameter(name="call_direction", value="inbound")
    response.append(connect)

    return HTMLResponse(content=str(response), media_type="application/xml")


# ─── Outbound Call Handler ───────────────────────────────────────

@app.get("/outbound")
async def outbound_call(phone_number: str):
    """
    Initiate an outbound call to the given phone number.
    The call connects to our WebSocket media stream for real-time AI conversation.
    """
    if not phone_number:
        return JSONResponse({"error": "phone_number is required"}, status_code=400)

    twiml = f"""
    <Response>
        <Say voice="Polly.Joanna">Hi there! How can I help you today?</Say>
        <Pause length="1"/>
        <Connect>
            <Stream url="wss://{get_ngrok_url()}/media-stream">
                <Parameter name="caller_number" value="{phone_number}"/>
                <Parameter name="call_direction" value="outbound"/>
            </Stream>
        </Connect>
    </Response>
    """

    try:
        call = twilio_client.calls.create(
            twiml=twiml.strip(),
            to=phone_number,
            from_=TWILIO_PHONE_NUMBER,
        )
        return JSONResponse({
            "message": f"Call initiated to {phone_number}",
            "call_sid": call.sid,
            "status": "initiated",
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ─── WebSocket Media Stream Handler ─────────────────────────────

@app.websocket("/media-stream")
async def media_stream(websocket: WebSocket):
    """
    Core WebSocket endpoint for Twilio Media Streams.
    Handles bidirectional audio streaming with real-time AI processing
    and barge-in support for natural conversations.

    Flow:
    1. Twilio sends raw audio → Deepgram STT (real-time transcription)
    2. Transcript → OpenAI LLM (streaming response)
    3. LLM text chunks → ElevenLabs TTS (streaming audio)
    4. TTS audio → Twilio (plays to caller)

    Barge-in: If the user speaks while AI is responding, the current
    LLM/TTS generation is cancelled and the new transcript is processed.
    """
    await websocket.accept()
    print("[WebSocket] Connection accepted")

    # State for this call session
    stream_sid = None
    caller_number = "unknown"
    call_direction = "unknown"
    call_start_time = datetime.utcnow()

    # Barge-in state
    current_processing_task = None
    cancel_event = asyncio.Event()

    # Initialize the LLM with conversation history
    llm = GroqLLM()

    # Queue to map mark names to streamSids for tracking
    mark_queue = {}

    # ── Callback: When TTS produces audio, send it to Twilio ──
    async def send_to_twilio(b64_audio: str):
        """Send audio chunk back to Twilio via the WebSocket."""
        nonlocal stream_sid
        if stream_sid and not cancel_event.is_set():
            try:
                message = {
                    "event": "media",
                    "streamSid": stream_sid, # Use the main stream_sid
                    "media": {
                        "payload": b64_audio,
                    },
                }
                await websocket.send_json(message)

                # Add Twilio Mark event so we know when playout finishes vs barge-in
                mark_name = f"mark_{int(time.time() * 1000)}"
                mark_queue[mark_name] = stream_sid # Store stream_sid with mark_name
                await websocket.send_json({
                    "event": "mark",
                    "streamSid": stream_sid, # Use the main stream_sid
                    "mark": {"name": mark_name}
                })
            except Exception:
                pass  # WebSocket may have closed

    # Global TTS instance (since it holds a long-lived ws session)
    global tts
    tts = SarvamTTS(send_to_twilio, language="hi-IN", speaker="shreya", model="bulbul:v3")

    # ── Process a transcript through the full pipeline ──
    async def process_transcript(transcript: str):
        """Process transcript through LLM → TTS → Twilio pipeline."""
        nonlocal cancel_event

        pipeline_start = time.time()
        print(f"[Pipeline] Processing: {transcript}")

        try:
            # Save user message to MongoDB
            await append_to_chat_history(caller_number, "user", transcript)

            # Stream LLM response → TTS
            full_response = ""
            first_chunk = True
            async for text_chunk in llm.stream_response(transcript, cancel_event):
                if cancel_event.is_set():
                    print("[Pipeline] Cancelled by barge-in")
                    return

                if first_chunk:
                    latency_ms = (time.time() - pipeline_start) * 1000
                    print(f"[Pipeline] First LLM chunk in {latency_ms:.0f}ms")
                    first_chunk = False

                await tts.send_text(text_chunk + " ")
                full_response += text_chunk

            if cancel_event.is_set():
                return

            # Flush remaining audio from TTS
            await tts.flush()

            # Save AI response to MongoDB
            if full_response:
                await append_to_chat_history(caller_number, "assistant", full_response)
                print(f"[Pipeline] Complete ({(time.time() - pipeline_start) * 1000:.0f}ms). "
                      f"Response: {full_response[:80]}...")

        except asyncio.CancelledError:
            print("[Pipeline] Task cancelled")
        except Exception as e:
            print(f"[Pipeline] Error: {e}")

    # ── Callback: When Deepgram produces a transcript, process it ──
    async def on_transcript(transcript: str):
        """Handle a final transcript — supports barge-in."""
        nonlocal current_processing_task, cancel_event, stream_sid

        if not transcript.strip():
            return

        # ── Barge-in: If currently processing, cancel and clear ──
        if current_processing_task and not current_processing_task.done():
            print(f"[Barge-in] Interrupting current response for: {transcript}")

            # 1. Signal cancellation to LLM stream
            cancel_event.set()

            # 2. Tell Twilio to stop playing queued audio
            if stream_sid:
                try:
                    clear_message = {
                        "event": "clear",
                        "streamSid": stream_sid,
                    }
                    await websocket.send_json(clear_message)
                except Exception:
                    pass

            # 3. Wait for old task to finish cleanup
            try:
                current_processing_task.cancel()
                await asyncio.shield(asyncio.sleep(0.05))
            except Exception:
                pass

            # 4. Clear and reconnect TTS to discard queued audio
            await tts.clear_and_reconnect()

        # Reset cancel event for new processing
        cancel_event = asyncio.Event()

        # Start processing the new transcript
        current_processing_task = asyncio.create_task(process_transcript(transcript))

    # Initialize STT
    stt = DeepgramSTT(on_transcript_callback=on_transcript)

    try:
        # Connect to external services in parallel for faster startup
        await asyncio.gather(
            stt.connect(),
            tts.connect(),
        )
        print("[WebSocket] All services connected — ready for audio")

        # ── Main message loop ──
        async for message in websocket.iter_text():
            data = json.loads(message)
            event = data.get("event")

            if event == "connected":
                print("[Twilio] Media stream connected")

            elif event == "start":
                start_data = data.get("start", {})
                stream_sid = start_data.get("streamSid")
                custom_params = start_data.get("customParameters", {})
                caller_number = custom_params.get("caller_number", "unknown")
                call_direction = custom_params.get("call_direction", "unknown")

                print(f"[Twilio] Stream started — SID: {stream_sid}")
                print(f"[Twilio] Caller: {caller_number}, Direction: {call_direction}")

                # Load existing chat history into LLM
                history = await get_chat_history(caller_number)
                if history:
                    llm.set_history(history[-10:])  # Last 10 messages for context
                    print(f"[MongoDB] Loaded {len(history)} history messages")

            elif event == "media":
                # Forward raw audio to Deepgram
                payload = data["media"]["payload"]
                audio_bytes = base64.b64decode(payload)
                await stt.send_audio(audio_bytes)

            elif event == "mark":
                # Track audio playback completion (useful for future enhancements)
                pass

            elif event == "stop":
                print("[Twilio] Media stream stopped")
                break

    except WebSocketDisconnect:
        print("[WebSocket] Client disconnected")
    except Exception as e:
        print(f"[WebSocket] Error: {e}")
    finally:
        # Cancel any in-progress processing
        if current_processing_task and not current_processing_task.done():
            cancel_event.set()
            current_processing_task.cancel()

        # Cleanup
        await stt.close()
        await tts.close()

        # Save call log
        call_end_time = datetime.utcnow()
        await save_call_log({
            "caller_number": caller_number,
            "direction": call_direction,
            "stream_sid": stream_sid,
            "start_time": call_start_time,
            "end_time": call_end_time,
            "duration_seconds": (call_end_time - call_start_time).total_seconds(),
        })

        # Save final conversation history
        history = llm.get_history()
        if history:
            await save_chat_history(caller_number, history)

        print("[WebSocket] Session cleaned up")


# ─── Run Server ──────────────────────────────────────────────────

def start_ngrok_tunnel():
    """Start ngrok tunnel if NGROK_AUTH_TOKEN is set and NGROK_URL is empty."""
    global _ngrok_domain
    if _ngrok_domain:
        print(f"[Ngrok] Using configured URL: {_ngrok_domain}")
        return
    if not NGROK_AUTH_TOKEN:
        print("[Ngrok] No NGROK_URL or NGROK_AUTH_TOKEN set — tunnel not started")
        return

    from pyngrok import ngrok, conf
    conf.get_default().auth_token = NGROK_AUTH_TOKEN
    tunnel = ngrok.connect(PORT)
    _ngrok_domain = tunnel.public_url.replace("https://", "").replace("http://", "")
    print(f"[Ngrok] Tunnel started: https://{_ngrok_domain}")
    print(f"[Ngrok] Set your Twilio webhook to: https://{_ngrok_domain}/inbound")


if __name__ == "__main__":
    import uvicorn
    start_ngrok_tunnel()
    uvicorn.run(app, host="0.0.0.0", port=PORT)
