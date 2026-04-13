"""
Text-to-Speech using Sarvam AI with WebSocket-first, REST-fallback strategy.

Tries the persistent WebSocket endpoint (wss://api.sarvam.ai/text-to-speech/ws)
for lowest latency.  If the WS connection fails or is unavailable, automatically
falls back to the REST API (text_to_speech.convert) which is slower but reliable.

The public interface (connect / send_text / flush / clear_and_reconnect / close)
stays the same regardless of which backend is active.
"""

import asyncio
import audioop
import base64
import json
import re
import time

import websockets
from sarvamai import AsyncSarvamAI
from config import SARVAM_API_KEY

SARVAM_WS_URL = "wss://api.sarvam.ai/text-to-speech/ws"

# Size of mulaw chunks sent to Twilio (4000 bytes ~ 500ms at 8kHz mulaw)
TWILIO_CHUNK_SIZE = 4000

# WebSocket connection timeout (seconds)
WS_CONNECT_TIMEOUT = 8


class SarvamTTS:
    """Streams text to Sarvam AI TTS and forwards mulaw audio to Twilio."""

    def __init__(self, on_audio, language="hi-IN", speaker="priya", model="bulbul:v3"):
        """
        Args:
            on_audio: async function called with (audio_base64: str) for each chunk.
            language: Target language code (e.g., 'hi-IN', 'en-IN').
            speaker: Voice name — bulbul:v3 compatible (e.g., 'priya', 'shreya').
            model: The TTS model to use ('bulbul:v3' recommended).
        """
        self.on_audio = on_audio
        self.language = language
        self.speaker = speaker
        self.model = model
        self._cancelled = False

        # WebSocket state
        self._ws = None
        self._listener_task: asyncio.Task | None = None
        self._flush_event = asyncio.Event()
        self._ws_connected = False

        # REST fallback state
        self._use_rest = False
        self._rest_client: AsyncSarvamAI | None = None
        self._text_buffer: list[str] = []

        # Latency tracking
        self._send_start_time: float | None = None
        self._first_audio_logged = False

    # ── Connection ──────────────────────────────────────────────

    async def connect(self):
        """Connect to Sarvam TTS — WebSocket first, fall back to REST."""
        # Try WebSocket first
        try:
            await self._connect_ws()
            return
        except Exception as e:
            print(f"[Sarvam TTS] WebSocket unavailable ({e}), falling back to REST API")

        # Fall back to REST
        self._use_rest = True
        self._rest_client = AsyncSarvamAI(api_subscription_key=SARVAM_API_KEY)
        print("[Sarvam TTS] REST API client ready (fallback mode)")

    async def _connect_ws(self):
        """Open WebSocket connection and send configuration."""
        url = f"{SARVAM_WS_URL}?model={self.model}&send_completion_event=true"
        extra_headers = {"Api-Subscription-Key": SARVAM_API_KEY}

        self._ws = await asyncio.wait_for(
            websockets.connect(url, additional_headers=extra_headers),
            timeout=WS_CONNECT_TIMEOUT,
        )
        self._ws_connected = True

        # Send configuration
        config_msg = {
            "type": "config",
            "data": {
                "target_language_code": self.language,
                "speaker": self.speaker,
                "pace": 1.0,
                "speech_sample_rate": "8000",
                "output_audio_codec": "mulaw",
            },
        }
        await self._ws.send(json.dumps(config_msg))

        # Start background listener
        self._listener_task = asyncio.create_task(self._listen_loop())

        # Brief wait to catch config rejection (0.5s needed for server round-trip)
        await asyncio.sleep(0.5)
        if not self._ws_connected:
            raise RuntimeError("Sarvam TTS config was rejected")

        self._use_rest = False
        print("[Sarvam TTS] WebSocket connected and configured")

    # ── WebSocket listener ──────────────────────────────────────

    async def _listen_loop(self):
        """Receive audio chunks from the Sarvam WebSocket."""
        try:
            async for message in self._ws:
                if self._cancelled:
                    continue

                try:
                    data = json.loads(message)
                except (json.JSONDecodeError, TypeError):
                    continue

                msg_type = data.get("type")

                if msg_type == "audio":
                    audio_b64 = data.get("data", {}).get("audio")
                    if not audio_b64:
                        continue

                    if not self._first_audio_logged and self._send_start_time:
                        latency_ms = (time.time() - self._send_start_time) * 1000
                        print(f"[Sarvam TTS] First audio in {latency_ms:.0f}ms")
                        self._first_audio_logged = True

                    mulaw_bytes = base64.b64decode(audio_b64)
                    for i in range(0, len(mulaw_bytes), TWILIO_CHUNK_SIZE):
                        if self._cancelled:
                            break
                        segment = mulaw_bytes[i : i + TWILIO_CHUNK_SIZE]
                        segment_b64 = base64.b64encode(segment).decode("utf-8")
                        await self.on_audio(segment_b64)

                elif msg_type == "event":
                    event_type = data.get("data", {}).get("event_type")
                    if event_type == "final":
                        self._flush_event.set()

                elif msg_type == "error":
                    err = data.get("data", {}).get("message", "unknown")
                    print(f"[Sarvam TTS] Server error: {err}")
                    self._ws_connected = False
                    self._flush_event.set()

        except websockets.ConnectionClosed:
            print("[Sarvam TTS] WebSocket closed")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            if not self._cancelled:
                print(f"[Sarvam TTS] Listener error: {e}")
        finally:
            self._ws_connected = False

    # ── Send text ───────────────────────────────────────────────

    async def send_text(self, text: str):
        """Send a text chunk for synthesis."""
        if self._cancelled or not text.strip():
            return

        # Skip purely punctuation/symbols
        if not re.search(r'[a-zA-Z\u0900-\u097F0-9]', text):
            return

        if self._send_start_time is None:
            self._send_start_time = time.time()
            self._first_audio_logged = False

        if self._use_rest:
            # Buffer text for REST API (synthesized on flush)
            self._text_buffer.append(text.strip())
            print(f"[Sarvam TTS] Buffered ({len(text.strip())} chars): {text.strip()[:60]}")
            return

        # WebSocket: stream text to server
        if not self._ws or not self._ws_connected:
            # WS died mid-session, buffer for REST fallback
            self._text_buffer.append(text.strip())
            return

        msg = {"type": "text", "data": {"text": text.strip()}}
        try:
            await self._ws.send(json.dumps(msg))
            print(f"[Sarvam TTS] Sent ({len(text.strip())} chars): {text.strip()[:60]}")
        except Exception as e:
            print(f"[Sarvam TTS] Send error: {e}, buffering for REST")
            self._text_buffer.append(text.strip())
            self._ws_connected = False
            self._use_rest = True
            if not self._rest_client:
                self._rest_client = AsyncSarvamAI(api_subscription_key=SARVAM_API_KEY)

    # ── Flush ───────────────────────────────────────────────────

    async def flush(self):
        """Synthesize/flush all pending text."""
        if self._cancelled:
            self._text_buffer.clear()
            return

        # If we have buffered text (REST mode or WS failed mid-stream)
        if self._text_buffer:
            await self._flush_rest()
            return

        # WebSocket flush
        if not self._ws or not self._ws_connected:
            return

        self._flush_event.clear()
        try:
            await self._ws.send(json.dumps({"type": "flush"}))
        except Exception as e:
            print(f"[Sarvam TTS] Flush send error: {e}")
            return

        try:
            await asyncio.wait_for(self._flush_event.wait(), timeout=15.0)
        except asyncio.TimeoutError:
            print("[Sarvam TTS] Flush timed out")

        self._send_start_time = None
        self._first_audio_logged = False

    async def _flush_rest(self):
        """Synthesize buffered text via REST API and stream audio to Twilio."""
        full_text = " ".join(self._text_buffer).strip()
        self._text_buffer.clear()

        if not full_text or not self._rest_client:
            return

        print(f"[Sarvam TTS] REST synthesizing ({len(full_text)} chars): {full_text[:60]}...")

        try:
            response = await self._rest_client.text_to_speech.convert(
                text=full_text,
                target_language_code=self.language,
                speaker=self.speaker,
                speech_sample_rate=8000,
                output_audio_codec="linear16",
                model=self.model,
                pace=1.0,
                enable_preprocessing=True,
            )

            if not response.audios or len(response.audios) == 0:
                print("[Sarvam TTS] REST API returned no audio")
                return

            if self._cancelled:
                return

            if not self._first_audio_logged and self._send_start_time:
                latency_ms = (time.time() - self._send_start_time) * 1000
                print(f"[Sarvam TTS] First audio in {latency_ms:.0f}ms (REST)")
                self._first_audio_logged = True

            # Decode linear16 PCM → mulaw for Twilio
            raw_pcm = base64.b64decode(response.audios[0])
            mulaw_bytes = audioop.lin2ulaw(raw_pcm, 2)

            # Stream in chunks to Twilio
            for i in range(0, len(mulaw_bytes), TWILIO_CHUNK_SIZE):
                if self._cancelled:
                    break
                chunk = mulaw_bytes[i : i + TWILIO_CHUNK_SIZE]
                mulaw_b64 = base64.b64encode(chunk).decode("utf-8")
                await self.on_audio(mulaw_b64)
                # Pace sending to avoid overwhelming Twilio
                await asyncio.sleep(len(chunk) / 8000.0 * 0.5)

        except Exception as e:
            print(f"[Sarvam TTS] REST synthesis error: {e}")

        self._send_start_time = None
        self._first_audio_logged = False

    # ── Barge-in / cleanup ──────────────────────────────────────

    async def clear_and_reconnect(self):
        """Barge-in: discard state and reconnect."""
        print("[Sarvam TTS] Barge-in: reconnecting")
        self._cancelled = True
        self._text_buffer.clear()

        await self._close_ws()

        self._cancelled = False
        self._flush_event.clear()
        self._send_start_time = None
        self._first_audio_logged = False

        # Try to reconnect (will fall back to REST if WS still down)
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
        self._ws_connected = False

    async def close(self):
        """Shutdown the TTS client."""
        self._cancelled = True
        self._text_buffer.clear()
        await self._close_ws()
        print("[Sarvam TTS] Disconnected")
