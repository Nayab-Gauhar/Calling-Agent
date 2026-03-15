"""
Real-time Text-to-Speech using Sarvam AI WebSocket Streaming API.
Connects to wss://api.sarvam.ai/text-to-speech/ws for persistent, low-latency
TTS synthesis.  Text is streamed in progressively from the LLM, and Sarvam
returns mulaw audio chunks that are forwarded directly to Twilio — no MP3
decoding step needed.

Key advantages over the HTTP streaming endpoint:
  - Persistent WebSocket (no per-request TCP/TLS overhead)
  - Native mulaw 8kHz output (no MP3→PCM→mulaw conversion)
  - Server-side text buffering with min_buffer_size / max_chunk_length
  - Flush + completion events for clean end-of-utterance handling
"""

import asyncio
import base64
import json
import time

import websockets
from config import SARVAM_API_KEY

SARVAM_WS_URL = "wss://api.sarvam.ai/text-to-speech/ws"

# Size of mulaw chunks sent to Twilio (4000 bytes ~ 500ms at 8kHz mulaw)
TWILIO_CHUNK_SIZE = 4000


class SarvamTTS:
    """Streams text to Sarvam AI TTS via WebSocket and forwards mulaw audio to Twilio."""

    def __init__(self, on_audio, language="hi-IN", speaker="shreya", model="bulbul:v3"):
        """
        Args:
            on_audio: async function called with (audio_base64: str) for each chunk.
            language: Target language code (e.g., 'hi-IN', 'en-IN').
            speaker: Voice name — must be bulbul:v3 compatible (e.g., 'shreya', 'shubh').
            model: The TTS model to use ('bulbul:v3' recommended).
        """
        self.on_audio = on_audio
        self.language = language
        self.speaker = speaker
        self.model = model
        self._cancelled = False
        self._ws = None
        self._listener_task: asyncio.Task | None = None
        self._flush_event = asyncio.Event()
        self._connected = False
        self._send_start_time: float | None = None  # tracks time-to-first-audio
        self._first_audio_logged = False

    async def connect(self):
        """Open WebSocket connection and send configuration."""
        url = f"{SARVAM_WS_URL}?model={self.model}&send_completion_event=true"
        extra_headers = {"Api-Subscription-Key": SARVAM_API_KEY}

        try:
            self._ws = await websockets.connect(url, additional_headers=extra_headers)
            self._connected = True

            # Send configuration message
            config_msg = {
                "type": "config",
                "data": {
                    "target_language_code": self.language,
                    "speaker": self.speaker,
                    "pace": 1.1,
                    "speech_sample_rate": "8000",
                    "output_audio_codec": "mulaw",
                    "min_buffer_size": 30,
                    "max_chunk_length": 150,
                },
            }
            await self._ws.send(json.dumps(config_msg))

            # Start background listener for audio chunks
            self._listener_task = asyncio.create_task(self._listen_loop())
            print("[Sarvam TTS] WebSocket connected and configured")

        except Exception as e:
            print(f"[Sarvam TTS] Connection error: {e}")
            self._connected = False
            raise

    async def _listen_loop(self):
        """
        Background task that receives messages from the Sarvam WebSocket.
        Audio chunks are decoded and forwarded to Twilio immediately.
        """
        try:
            async for message in self._ws:
                if self._cancelled:
                    continue

                try:
                    data = json.loads(message)
                except (json.JSONDecodeError, TypeError):
                    # Binary message or parse error — skip
                    continue

                msg_type = data.get("type")

                if msg_type == "audio":
                    audio_data = data.get("data", {})
                    audio_b64 = audio_data.get("audio")
                    if not audio_b64:
                        continue

                    if not self._first_audio_logged and self._send_start_time:
                        latency_ms = (time.time() - self._send_start_time) * 1000
                        print(f"[Sarvam TTS] First audio in {latency_ms:.0f}ms")
                        self._first_audio_logged = True

                    # Decode the base64 mulaw audio from Sarvam
                    mulaw_bytes = base64.b64decode(audio_b64)

                    # Forward to Twilio in appropriately-sized chunks
                    for i in range(0, len(mulaw_bytes), TWILIO_CHUNK_SIZE):
                        if self._cancelled:
                            break
                        segment = mulaw_bytes[i : i + TWILIO_CHUNK_SIZE]
                        segment_b64 = base64.b64encode(segment).decode("utf-8")
                        await self.on_audio(segment_b64)

                elif msg_type == "event":
                    event_data = data.get("data", {})
                    event_type = event_data.get("event_type")
                    if event_type == "final":
                        # Utterance synthesis complete
                        self._flush_event.set()

                elif msg_type == "error":
                    err = data.get("data", {}).get("message", "unknown")
                    print(f"[Sarvam TTS] Server error: {err}")

        except websockets.ConnectionClosed:
            print("[Sarvam TTS] WebSocket closed")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            if not self._cancelled:
                print(f"[Sarvam TTS] Listener error: {e}")
        finally:
            self._connected = False

    async def send_text(self, text: str):
        """
        Send a text chunk to Sarvam for synthesis.
        Text is buffered server-side according to min_buffer_size / max_chunk_length.
        """
        if self._cancelled or not text.strip() or not self._ws or not self._connected:
            return

        if self._send_start_time is None:
            self._send_start_time = time.time()
            self._first_audio_logged = False

        msg = {"type": "text", "data": {"text": text.strip()}}
        try:
            await self._ws.send(json.dumps(msg))
            print(f"[Sarvam TTS] Sent ({len(text.strip())} chars): {text.strip()[:60]}")
        except Exception as e:
            print(f"[Sarvam TTS] Send error: {e}")

    async def flush(self):
        """
        Signal end-of-utterance.  Sends a flush command to Sarvam and waits
        for the completion event so all audio is delivered before returning.
        """
        if self._cancelled or not self._ws or not self._connected:
            return

        self._flush_event.clear()

        try:
            await self._ws.send(json.dumps({"type": "flush"}))
        except Exception as e:
            print(f"[Sarvam TTS] Flush send error: {e}")
            return

        # Wait for the "final" completion event (with timeout)
        try:
            await asyncio.wait_for(self._flush_event.wait(), timeout=15.0)
        except asyncio.TimeoutError:
            print("[Sarvam TTS] Flush timed out waiting for completion event")

        # Reset timing for next utterance
        self._send_start_time = None
        self._first_audio_logged = False

    async def clear_and_reconnect(self):
        """Barge-in: close the current WebSocket and open a fresh one."""
        print("[Sarvam TTS] Barge-in: reconnecting")
        self._cancelled = True

        # Close existing connection
        await self._close_ws()

        # Reset state
        self._cancelled = False
        self._flush_event.clear()
        self._send_start_time = None
        self._first_audio_logged = False

        # Reconnect
        try:
            await self.connect()
        except Exception as e:
            print(f"[Sarvam TTS] Reconnect failed: {e}")

    async def _close_ws(self):
        """Close the WebSocket and listener task."""
        if self._listener_task:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except (asyncio.CancelledError, Exception):
                pass
            self._listener_task = None

        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
        self._connected = False

    async def close(self):
        """Shutdown the TTS client."""
        self._cancelled = True
        await self._close_ws()
        print("[Sarvam TTS] Disconnected")
