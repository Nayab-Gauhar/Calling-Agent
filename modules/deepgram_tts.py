"""
Real-time Text-to-Speech using Deepgram's Aura WebSocket streaming API.
Accepts text chunks and yields audio bytes suitable for Twilio playback.
Includes auto-reconnect and buffer clearing for barge-in support.
Compatible with websockets v16+.
"""

import json
import asyncio
import base64
import websockets
from config import DEEPGRAM_API_KEY

DEEPGRAM_TTS_WS_URL = "wss://api.deepgram.com/v1/speak"
MAX_RECONNECT_ATTEMPTS = 5
RECONNECT_DELAY_S = 1.0


class DeepgramTTS:
    """Streams text to Deepgram Aura TTS and receives audio chunks in real-time."""

    def __init__(self, on_audio_callback, model="aura-2-thalia-en", encoding="mulaw", sample_rate=8000):
        """
        Args:
            on_audio_callback: async function called with (audio_base64: str) for each
                               chunk of audio received. Audio is base64-encoded mulaw.
            model: Deepgram TTS voice model.
            encoding: Audio encoding (mulaw for Twilio).
            sample_rate: Audio sample rate (8000 for Twilio).
        """
        self.on_audio = on_audio_callback
        self.model = model
        self.encoding = encoding
        self.sample_rate = sample_rate
        self.ws = None
        self._listen_task = None
        self._connected = False
        self._reconnect_attempts = 0

    def _build_url(self) -> str:
        """Build the Deepgram TTS WebSocket URL."""
        return (
            f"{DEEPGRAM_TTS_WS_URL}"
            f"?model={self.model}"
            f"&encoding={self.encoding}"
            f"&sample_rate={self.sample_rate}"
        )

    def _is_open(self) -> bool:
        """Check if the WebSocket connection is open (compatible with websockets v16+)."""
        if self.ws is None:
            return False
        try:
            from websockets.protocol import State
            return self.ws.state is State.OPEN
        except (ImportError, AttributeError):
            pass
        try:
            return self.ws.open
        except AttributeError:
            return self.ws is not None

    async def connect(self):
        """Establish WebSocket connection to Deepgram TTS."""
        url = self._build_url()
        headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}"}

        self.ws = await websockets.connect(
            url,
            additional_headers=headers,
            ping_interval=20,
            ping_timeout=10,
        )
        self._connected = True
        self._reconnect_attempts = 0
        print("[Deepgram TTS] Connected to streaming TTS")

        # Start listening for audio chunks
        self._listen_task = asyncio.create_task(self._listen())

    async def send_text(self, text: str):
        """Send a text chunk to Deepgram for synthesis."""
        if self._connected and self._is_open():
            try:
                message = json.dumps({
                    "type": "Speak",
                    "text": text,
                })
                await self.ws.send(message)
            except websockets.exceptions.ConnectionClosed:
                self._connected = False
                await self._reconnect()

    async def flush(self):
        """Signal to flush queued text and get the audio."""
        if self._connected and self._is_open():
            try:
                await self.ws.send(json.dumps({"type": "Flush"}))
            except websockets.exceptions.ConnectionClosed:
                self._connected = False

    async def clear_and_reconnect(self):
        """
        Clear queued audio by closing and reopening the connection.
        Used for barge-in — stops any audio that hasn't been sent yet.
        """
        print("[Deepgram TTS] Clearing buffer (barge-in) — reconnecting")
        await self._force_close()
        try:
            await self.connect()
        except Exception as e:
            print(f"[Deepgram TTS] Reconnect after clear failed: {e}")

    async def _listen(self):
        """Listen for audio chunks from Deepgram TTS."""
        try:
            async for message in self.ws:
                if isinstance(message, bytes):
                    # Binary frame = raw audio data
                    # Encode to base64 for Twilio
                    audio_base64 = base64.b64encode(message).decode("utf-8")
                    await self.on_audio(audio_base64)
                elif isinstance(message, str):
                    # JSON metadata messages
                    try:
                        data = json.loads(message)
                        msg_type = data.get("type", "")
                        if msg_type == "Flushed":
                            pass  # Flush complete
                        elif msg_type == "Metadata":
                            pass  # Metadata info
                        elif msg_type == "Warning":
                            print(f"[Deepgram TTS] Warning: {data.get('warn_msg', '')}")
                    except json.JSONDecodeError:
                        pass

        except websockets.exceptions.ConnectionClosed:
            print("[Deepgram TTS] Connection closed")
            self._connected = False
            await self._reconnect()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"[Deepgram TTS] Error in listener: {e}")
            self._connected = False
            await self._reconnect()

    async def _reconnect(self):
        """Attempt to reconnect with exponential backoff."""
        if self._reconnect_attempts >= MAX_RECONNECT_ATTEMPTS:
            print("[Deepgram TTS] Max reconnect attempts reached, giving up")
            return

        self._reconnect_attempts += 1
        delay = RECONNECT_DELAY_S * self._reconnect_attempts
        print(f"[Deepgram TTS] Reconnecting in {delay}s (attempt {self._reconnect_attempts})")
        await asyncio.sleep(delay)

        try:
            await self.connect()
            print("[Deepgram TTS] Reconnected successfully")
        except Exception as e:
            print(f"[Deepgram TTS] Reconnect failed: {e}")
            await self._reconnect()

    async def _force_close(self):
        """Force close the WebSocket without sending Close."""
        self._connected = False
        if self._listen_task:
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass
        if self.ws:
            try:
                await self.ws.close()
            except Exception:
                pass

    async def close(self):
        """Close the Deepgram TTS WebSocket connection gracefully."""
        self._connected = False
        if self._listen_task:
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass
        if self._is_open():
            try:
                await self.ws.send(json.dumps({"type": "Close"}))
            except Exception:
                pass
            try:
                await self.ws.close()
            except Exception:
                pass
            print("[Deepgram TTS] Disconnected")
