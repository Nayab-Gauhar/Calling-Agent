"""
Real-time Speech-to-Text using Deepgram's WebSocket streaming API.
Receives raw audio from Twilio Media Streams and returns transcripts.
Includes auto-reconnect and optimized VAD for low latency.
Supports multilingual (Hindi + English) recognition.
Compatible with websockets v16+.
"""

import json
import asyncio
import websockets
from config import DEEPGRAM_API_KEY, DEEPGRAM_ENDPOINTING_MS, DEEPGRAM_LANGUAGE

DEEPGRAM_WS_URL = "wss://api.deepgram.com/v1/listen"
MAX_RECONNECT_ATTEMPTS = 5
RECONNECT_DELAY_S = 1.0


class DeepgramSTT:
    """Manages a persistent WebSocket connection to Deepgram for real-time STT."""

    def __init__(self, on_transcript_callback, language=None, sample_rate=8000, encoding="mulaw", channels=1):
        """
        Args:
            on_transcript_callback: async function called with (transcript: str) when
                                    a final transcript is received.
            language: Primary language for recognition. Use "hi" for Hindi,
                      "multi" for automatic multi-language detection,
                      or "en" for English-only. Defaults to config value.
            sample_rate: Audio sample rate (Twilio sends 8000 Hz).
            encoding: Audio encoding (Twilio sends mulaw).
            channels: Number of audio channels.
        """
        self.on_transcript = on_transcript_callback
        self.language = language or DEEPGRAM_LANGUAGE
        self.sample_rate = sample_rate
        self.encoding = encoding
        self.channels = channels
        self.ws = None
        self._listen_task = None
        self._reconnect_task = None
        self._connected = False
        self._reconnect_attempts = 0

    def _build_url(self) -> str:
        """Build the Deepgram WebSocket URL with optimized parameters."""
        params = (
            f"?encoding={self.encoding}"
            f"&sample_rate={self.sample_rate}"
            f"&channels={self.channels}"
            f"&language={self.language}"
            f"&model=nova-3"
            f"&smart_format=true"
            f"&filler_words=false"
            f"&punctuate=true"
            f"&interim_results=true"
            f"&endpointing={DEEPGRAM_ENDPOINTING_MS}"
            f"&utterance_end_ms=1500"
            f"&vad_events=true"
        )
        return DEEPGRAM_WS_URL + params

    def _is_open(self) -> bool:
        """Check if the WebSocket connection is open (compatible with websockets v16+)."""
        if self.ws is None:
            return False
        try:
            # websockets v16+: use state
            from websockets.protocol import State
            return self.ws.state is State.OPEN
        except (ImportError, AttributeError):
            pass
        try:
            # Fallback: try .open attribute (older versions)
            return self.ws.open
        except AttributeError:
            # Final fallback: assume open if ws exists
            return self.ws is not None

    async def connect(self):
        """Establish WebSocket connection to Deepgram."""
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
        print("[Deepgram] Connected to streaming STT")

        # Start listening for transcripts in background
        self._listen_task = asyncio.create_task(self._listen())

    async def send_audio(self, audio_bytes: bytes):
        """Send raw audio bytes to Deepgram."""
        if self._connected and self._is_open():
            try:
                await self.ws.send(audio_bytes)
            except websockets.exceptions.ConnectionClosed:
                self._connected = False
                # Track reconnect task to prevent concurrent attempts
                if not self._reconnect_task or self._reconnect_task.done():
                    self._reconnect_task = asyncio.create_task(self._reconnect())

    async def _listen(self):
        """
        Listen for transcript results from Deepgram.
        
        Strategy: Accumulate is_final transcripts and only dispatch to the
        callback when speech_final=True OR an UtteranceEnd event arrives.
        This prevents cutting off Hindi/Hinglish speakers mid-sentence
        (where pauses between clauses trigger is_final but the person
        hasn't finished their thought yet).
        """
        accumulated = ""
        try:
            async for message in self.ws:
                data = json.loads(message)
                msg_type = data.get("type")

                if msg_type == "Results":
                    channel = data.get("channel", {})
                    alternatives = channel.get("alternatives", [])

                    if alternatives:
                        transcript = alternatives[0].get("transcript", "").strip()
                        is_final = data.get("is_final", False)
                        speech_final = data.get("speech_final", False)

                        if transcript and is_final:
                            accumulated += (" " + transcript) if accumulated else transcript

                        # speech_final = Deepgram is confident the speaker finished
                        if speech_final and accumulated.strip():
                            full = accumulated.strip()
                            accumulated = ""
                            print(f"[Deepgram] Final transcript: {full}")
                            await self.on_transcript(full)

                elif msg_type == "UtteranceEnd":
                    # Backup: if speech_final didn't fire but silence exceeded
                    # utterance_end_ms, flush whatever we have
                    if accumulated.strip():
                        full = accumulated.strip()
                        accumulated = ""
                        print(f"[Deepgram] UtteranceEnd transcript: {full}")
                        await self.on_transcript(full)

        except websockets.exceptions.ConnectionClosed:
            print("[Deepgram] Connection closed")
            self._connected = False
            if not self._reconnect_task or self._reconnect_task.done():
                self._reconnect_task = asyncio.create_task(self._reconnect())
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"[Deepgram] Error in listener: {e}")
            self._connected = False
            if not self._reconnect_task or self._reconnect_task.done():
                self._reconnect_task = asyncio.create_task(self._reconnect())

    async def _reconnect(self):
        """Attempt to reconnect to Deepgram with exponential backoff."""
        if self._reconnect_attempts >= MAX_RECONNECT_ATTEMPTS:
            print("[Deepgram] Max reconnect attempts reached, giving up")
            return

        self._reconnect_attempts += 1
        delay = RECONNECT_DELAY_S * self._reconnect_attempts
        print(f"[Deepgram] Reconnecting in {delay}s (attempt {self._reconnect_attempts})")
        await asyncio.sleep(delay)

        try:
            await self.connect()
            print("[Deepgram] Reconnected successfully")
        except Exception as e:
            print(f"[Deepgram] Reconnect failed: {e}")
            await self._reconnect()

    async def close(self):
        """Close the Deepgram WebSocket connection."""
        self._connected = False
        if self._listen_task:
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass
        if self._is_open():
            try:
                await self.ws.send(json.dumps({"type": "CloseStream"}))
                await self.ws.close()
            except Exception:
                pass
            print("[Deepgram] Disconnected")
