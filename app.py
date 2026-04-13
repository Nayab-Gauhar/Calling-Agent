"""
AI Calling Agent — Real-Time Voice Agent
Uses FastAPI + Twilio Media Streams + Deepgram STT + Groq LLM + Sarvam TTS
for human-like, low-latency phone conversations with barge-in support.
"""

import json
import base64
import asyncio
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
)
from modules.deepgram_stt import DeepgramSTT
from modules.nvidia_llm import GroqLLM
from modules.sarvam_tts import SarvamTTS
from modules.telegram import send_call_summary
from modules.call_summary import generate_call_summary
from modules.google_sheets import (
    log_call as log_call_to_sheets,
    get_chat_history,
    append_to_chat_history,
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

    # Connect to bidirectional media stream immediately.
    # The LLM generates the greeting naturally based on system_prompt.txt.
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

    # Track the greeting task so barge-in can cancel it
    greeting_task = None

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
            except Exception:
                pass  # WebSocket may have closed

    # TTS instance scoped to this session (not global to avoid concurrency issues)
    tts = SarvamTTS(send_to_twilio, language="en-IN", speaker="priya", model="bulbul:v3")

    # ── Process a transcript through the full pipeline ──
    async def process_transcript(transcript: str):
        """Process transcript through LLM → TTS → Twilio pipeline."""
        nonlocal cancel_event

        pipeline_start = time.time()
        print(f"[Pipeline] Processing: {transcript}")

        try:
            # Save user message to MongoDB
            await append_to_chat_history(caller_number, "user", transcript)

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

                # Flush at sentence boundaries for lower first-audio latency.
                # Only in WebSocket mode — in REST fallback, batching into one
                # call is faster than multiple 5-8s REST calls per sentence.
                last_char = text_chunk.rstrip()[-1:]
                if last_char in {'.', '!', '?', '।'} and not tts._use_rest:
                    await tts.flush()

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
        nonlocal current_processing_task, cancel_event, stream_sid, greeting_task

        if not transcript.strip():
            return

        # ── Barge-in: cancel greeting or active pipeline ──
        active_task = greeting_task if (greeting_task and not greeting_task.done()) else current_processing_task
        if active_task and not active_task.done():
            # Filter out very short/accidental barge-ins (e.g. echo of "Hello", coughs, "ok")
            cleaned_transcript = transcript.strip().lower().replace(".", "").replace("?", "").replace("!", "")
            ignore_phrases = ["hello", "yeah", "ok", "okay", "yes", "hmm", "hm", "oh", "ha", "haan", "hi"]
            words = cleaned_transcript.split()
            if len(words) <= 2 and all(w in ignore_phrases for w in words):
                print(f"[Barge-in] Ignored short/echo transcript during generation: {transcript}")
                return

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
            active_task.cancel()
            try:
                await asyncio.wait_for(
                    asyncio.shield(active_task), timeout=1.0
                )
            except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
                pass

            # Clear greeting ref if that's what we interrupted
            if active_task is greeting_task:
                greeting_task = None

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
                    print(f"[Google Sheets] Loaded {len(history)} history messages")

                # Generate greeting in a background task so the message loop
                # keeps processing audio immediately (prevents buffering caller's
                # early speech while the LLM generates the greeting).
                async def _generate_greeting():
                    try:
                        print("[Pipeline] Generating greeting via LLM")
                        greet = ""
                        async for text_chunk in llm.stream_greeting(cancel_event):
                            if cancel_event.is_set():
                                break
                            await tts.send_text(text_chunk + " ")
                            greet += text_chunk
                        if not cancel_event.is_set():
                            await tts.flush()
                        if greet:
                            await append_to_chat_history(caller_number, "assistant", greet)
                            print(f"[Pipeline] Greeting: {greet[:80]}...")
                    except asyncio.CancelledError:
                        print("[Pipeline] Greeting cancelled (barge-in)")
                    except Exception as e:
                        print(f"[Pipeline] Greeting error: {e}")

                greeting_task = asyncio.create_task(_generate_greeting())

            elif event == "media":
                # Forward raw audio to Deepgram
                payload = data["media"]["payload"]
                audio_bytes = base64.b64decode(payload)
                await stt.send_audio(audio_bytes)

            elif event == "mark":
                pass  # Mark events not currently used

            elif event == "stop":
                print("[Twilio] Media stream stopped")
                break

    except WebSocketDisconnect:
        print("[WebSocket] Client disconnected")
    except Exception as e:
        print(f"[WebSocket] Error: {e}")
    finally:
        # Cancel any in-progress processing
        cancel_event.set()
        for task in [greeting_task, current_processing_task]:
            if task and not task.done():
                task.cancel()

        # Cleanup
        await stt.close()
        await tts.close()

        # Gather call data
        call_end_time = datetime.now(timezone.utc)
        history = llm.get_history()

        # Send call summary to Telegram
        call_duration = (call_end_time - call_start_time).total_seconds()

        # Generate AI summary using NVIDIA LLM
        ai_summary = None
        try:
            ai_summary = await generate_call_summary(history or [])
        except Exception as e:
            print(f"[Summary] Failed: {e}")

        # Log to Google Sheets
        try:
            await log_call_to_sheets(
                caller_number=caller_number,
                direction=call_direction,
                duration_seconds=call_duration,
                summary=ai_summary,
            )
        except Exception as e:
            print(f"[Google Sheets] Failed: {e}")

        # Send to Telegram
        try:
            sent = await send_call_summary(
                caller_number=caller_number,
                direction=call_direction,
                stream_sid=stream_sid,
                duration_seconds=call_duration,
                conversation=history or [],
            )
            print(f"[Telegram] Call summary sent: {sent} ({len(history or [])} messages)")
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
