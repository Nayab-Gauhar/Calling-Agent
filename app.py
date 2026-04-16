"""
AI Calling Agent — Real-Time Voice Agent
Uses FastAPI + Twilio Media Streams + Deepgram STT + Groq LLM + Sarvam TTS
for human-like, low-latency phone conversations with barge-in support.
"""

import json
import base64
import asyncio
import random
import time
from datetime import datetime, timezone
from xml.sax.saxutils import escape as xml_escape

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
from modules.telegram import send_call_summary
from modules.google_sheets import append_call_log

app = FastAPI(title="AI Calling Agent", version="2.0.0")
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Serve static files (for any pre-recorded audio like greetings)
app.mount("/static", StaticFiles(directory="static"), name="static")

# ─── Filler sounds to fill dead air while LLM is thinking ────────
# These are short Hindi conversational fillers that a real person would say
# while processing what the caller said. Sent to TTS immediately before LLM
# starts generating, so the caller hears something within ~200ms.
FILLERS = [
    "हम्म",
    "हाँ",
    "अच्छा",
    "जी",
    "सही",
    "हाँ हाँ",
    "ओह",
    "अरे",
    "हम्म, देखो",
    "अच्छा अच्छा",
]

# Greeting sent through our own TTS pipeline when the call starts.
# This replaces the robotic Twilio Polly greeting.
INITIAL_GREETING = "हेलो! कैसे हो? बोलो, क्या बात है?"

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

    # Connect to bidirectional media stream immediately — no Polly greeting.
    # Our own TTS pipeline sends a natural Hindi greeting once the stream starts.
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

    safe_number = xml_escape(phone_number, {'"': "&quot;"})
    twiml = f"""
    <Response>
        <Connect>
            <Stream url="wss://{get_ngrok_url()}/media-stream">
                <Parameter name="caller_number" value="{safe_number}"/>
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
    2. Transcript → Groq LLM (streaming response)
    3. LLM text chunks → Sarvam TTS (streaming audio)
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
    call_start_time = datetime.now(timezone.utc)

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

    # TTS instance scoped to this session (not global to avoid concurrency issues)
    tts = SarvamTTS(send_to_twilio, language="hi-IN", speaker="shreya", model="bulbul:v3")

    # ── Process a transcript through the full pipeline ──
    async def process_transcript(transcript: str):
        """Process transcript through filler → LLM → TTS → Twilio pipeline."""
        nonlocal cancel_event

        pipeline_start = time.time()
        print(f"[Pipeline] Processing: {transcript}")

        try:
            # Save user message to MongoDB
            await append_to_chat_history(caller_number, "user", transcript)

            # ── INSTANT FILLER: Send a short filler to TTS immediately ──
            # This fills the dead air while the LLM is thinking (~300-500ms)
            # so the caller hears "हम्म" or "हाँ" instead of silence.
            filler = random.choice(FILLERS)
            print(f"[Pipeline] Sending filler: {filler}")
            await tts.send_text(filler)
            await tts.flush()

            if cancel_event.is_set():
                return

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

            # 3. Cancel the task and WAIT for it to actually finish
            #    (prevents two concurrent pipelines sending interleaved audio)
            current_processing_task.cancel()
            try:
                await asyncio.wait_for(
                    asyncio.shield(current_processing_task), timeout=1.0
                )
            except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
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

                # Send initial greeting through our own TTS pipeline
                # (replaces the robotic Twilio Polly.Joanna greeting)
                print(f"[Pipeline] Sending initial greeting via TTS")
                await tts.send_text(INITIAL_GREETING)
                await tts.flush()
                # Add greeting to LLM history so it knows what it said
                llm.add_message("assistant", INITIAL_GREETING)
                await append_to_chat_history(caller_number, "assistant", INITIAL_GREETING)

            elif event == "media":
                # Forward raw audio to Deepgram
                payload = data["media"]["payload"]
                audio_bytes = base64.b64decode(payload)
                await stt.send_audio(audio_bytes)

            elif event == "mark":
                # Clean up completed marks to prevent unbounded growth
                mark_name = data.get("mark", {}).get("name")
                mark_queue.pop(mark_name, None)

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
        call_end_time = datetime.now(timezone.utc)
        call_log_data = {
            "caller_number": caller_number,
            "direction": call_direction,
            "stream_sid": stream_sid,
            "start_time": call_start_time,
            "end_time": call_end_time,
            "duration_seconds": (call_end_time - call_start_time).total_seconds(),
        }
        await save_call_log(call_log_data)
        
        # Append call log to Google Sheets with extracted data
        try:
            await append_call_log(call_log_data, conversation=history)
        except Exception as e:
            print(f"[Google Sheets] Error appending call log: {e}")

        # Save final conversation history
        history = llm.get_history()
        if history:
            await save_chat_history(caller_number, history)

        # Send call summary to Telegram
        call_duration = (call_end_time - call_start_time).total_seconds()
        try:
            await send_call_summary(
                caller_number=caller_number,
                direction=call_direction,
                stream_sid=stream_sid,
                duration_seconds=call_duration,
                conversation=history or [],
            )
        except Exception as e:
            print(f"[Telegram] Failed to send call summary: {e}")

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
