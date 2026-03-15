"""
Real-time Text-to-Speech using ElevenLabs WebSocket streaming API.
Accepts text chunks and yields audio bytes suitable for Twilio playback.
Includes auto-reconnect and buffer clearing for barge-in support.
Compatible with websockets v16+.
"""

import json
import asyncio
import websockets
from config import ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID

ELEVENLABS_WS_URL = "wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input"
MAX_RECONNECT_ATTEMPTS = 5
RECONNECT_DELAY_S = 1.0


class ElevenLabsTTS:
    """Streams text to ElevenLabs and receives audio chunks in real-time."""

    def __init__(self, on_audio_callback, voice_id=None, model_id="eleven_flash_v2_5"):
        """
        Args:
            on_audio_callback: async function called with (audio_base64: str) for each
                               chunk of audio received. Audio is base64-encoded mulaw.
            voice_id: ElevenLabs voice ID. Defaults to config value.
            model_id: ElevenLabs model. eleven_flash_v2_5 for lowest latency.
        """
        self.on_audio = on_audio_callback
        self.voice_id = voice_id or ELEVENLABS_VOICE_ID
        self.model_id = model_id
        self.ws = None
        self._listen_task = None
        self._connected = False
        self._reconnect_attempts = 0

    def _build_url(self) -> str:
        """Build the ElevenLabs WebSocket URL."""
        url = ELEVENLABS_WS_URL.format(voice_id=self.voice_id)
        url += f"?model_id={self.model_id}&output_format=ulaw_8000"
        return url

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
        """Establish WebSocket connection to ElevenLabs."""
        url = self._build_url()

        self.ws = await websockets.connect(
            url,
            ping_interval=20,
            ping_timeout=10,
        )

        # Send initial config message (BOS - Beginning of Stream)
        bos_message = {
            "text": " ",
            "voice_settings": {
                "stability": 0.4,
                "similarity_boost": 0.75,
                "speed": 1.0,
            },
            "xi_api_key": ELEVENLABS_API_KEY,
            "generation_config": {
                "chunk_length_schedule": [50],
            },
        }
        await self.ws.send(json.dumps(bos_message))
        self._connected = True
        self._reconnect_attempts = 0
        print("[ElevenLabs] Connected to streaming TTS")

        # Start listening for audio chunks
        self._listen_task = asyncio.create_task(self._listen())

    async def send_text(self, text: str):
        """Send a text chunk to ElevenLabs for synthesis."""
        if self._connected and self._is_open():
            try:
                message = {
                    "text": text,
                    "try_trigger_generation": True,
                }
                await self.ws.send(json.dumps(message))
            except websockets.exceptions.ConnectionClosed:
                self._connected = False
                await self._reconnect()

    async def flush(self):
        """Signal end of text input to flush remaining audio."""
        if self._connected and self._is_open():
            try:
                message = {"text": ""}
                await self.ws.send(json.dumps(message))
            except websockets.exceptions.ConnectionClosed:
                self._connected = False

    async def clear_and_reconnect(self):
        """
        Clear queued audio by closing and reopening the connection.
        Used for barge-in — stops any audio that hasn't been sent yet.
        """
        print("[ElevenLabs] Clearing buffer (barge-in) — reconnecting")
        await self._force_close()
        try:
            await self.connect()
        except Exception as e:
            print(f"[ElevenLabs] Reconnect after clear failed: {e}")

    async def _listen(self):
        """Listen for audio chunks from ElevenLabs."""
        try:
            async for message in self.ws:
                data = json.loads(message)

                if data.get("audio"):
                    audio_base64 = data["audio"]
                    await self.on_audio(audio_base64)

                if data.get("isFinal"):
                    pass  # Segment complete

        except websockets.exceptions.ConnectionClosed:
            print("[ElevenLabs] Connection closed")
            self._connected = False
            await self._reconnect()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"[ElevenLabs] Error in listener: {e}")
            self._connected = False
            await self._reconnect()

    async def _reconnect(self):
        """Attempt to reconnect with exponential backoff."""
        if self._reconnect_attempts >= MAX_RECONNECT_ATTEMPTS:
            print("[ElevenLabs] Max reconnect attempts reached, giving up")
            return

        self._reconnect_attempts += 1
        delay = RECONNECT_DELAY_S * self._reconnect_attempts
        print(f"[ElevenLabs] Reconnecting in {delay}s (attempt {self._reconnect_attempts})")
        await asyncio.sleep(delay)

        try:
            await self.connect()
            print("[ElevenLabs] Reconnected successfully")
        except Exception as e:
            print(f"[ElevenLabs] Reconnect failed: {e}")
            await self._reconnect()

    async def _force_close(self):
        """Force close the WebSocket without sending EOS."""
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
        """Close the ElevenLabs WebSocket connection gracefully."""
        self._connected = False
        if self._listen_task:
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass
        if self._is_open():
            try:
                await self.ws.send(json.dumps({"text": ""}))
            except Exception:
                pass
            try:
                await self.ws.close()
            except Exception:
                pass
            print("[ElevenLabs] Disconnected")
