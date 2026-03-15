"""
Real-time Text-to-Speech using Sarvam AI.
Uses the official sarvamai SDK's AsyncSarvamAI client to open WebSocket sessions,
then manually reads audio chunks to feed them to Twilio.
Strategy: buffer all LLM text, then open one SDK session per utterance.
"""

import asyncio
import audioop
import base64
import io
import json
import array

import miniaudio
from sarvamai import AsyncSarvamAI
from config import SARVAM_API_KEY

_client = AsyncSarvamAI(api_subscription_key=SARVAM_API_KEY)


def mp3_b64_to_mulaw_b64(mp3_b64: str) -> str:
    """Convert base64-encoded MP3 to base64 mulaw 8000Hz mono (for Twilio)."""
    mp3_bytes = base64.b64decode(mp3_b64)

    # Decode MP3 → 16-bit PCM (at original sample rate / channels)
    decoded = miniaudio.mp3_read_s16(mp3_bytes)
    pcm_bytes = bytes(decoded.samples)
    src_rate = decoded.sample_rate
    src_channels = decoded.nchannels

    # Downmix to mono if stereo
    if src_channels == 2:
        pcm_bytes = audioop.tomono(pcm_bytes, 2, 0.5, 0.5)

    # Resample to 8000Hz
    pcm_bytes, _ = audioop.ratecv(pcm_bytes, 2, 1, src_rate, 8000, None)

    # Convert 16-bit PCM → mulaw
    mulaw_bytes = audioop.lin2ulaw(pcm_bytes, 2)
    return base64.b64encode(mulaw_bytes).decode("utf-8")


class SarvamTTS:
    """Streams text to Sarvam AI TTS and receives audio chunks in real-time."""

    def __init__(self, on_audio, language="hi-IN", speaker="shreya", model="bulbul:v3"):
        """
        Args:
            on_audio: async function called with (audio_base64: str) for each chunk.
            language: Target language code (e.g., 'hi-IN', 'en-IN').
            speaker: Voice name (e.g., 'shreya', 'arya', 'abhilash').
            model: The TTS model to use (e.g., 'bulbul:v3').
        """
        self.on_audio = on_audio
        self.language = language
        self.speaker = speaker
        self.model = model
        self._text_buffer = []  # Buffer for text chunks
        self._active_ws = None  # Reference to the active SDK websocket (for cancellation)
        self._cancelled = False

    async def connect(self):
        """No-op: sessions are opened fresh per utterance."""
        pass

    async def send_text(self, text: str):
        """Buffer a text chunk — will be synthesized all at once on flush()."""
        if not self._cancelled and text.strip():
            self._text_buffer.append(text.strip())

    async def flush(self):
        """
        Synthesize all buffered text in one Sarvam SDK session.
        Blocks until all audio has been delivered to the callback.
        """
        if self._cancelled:
            self._text_buffer.clear()
            return

        full_text = " ".join(self._text_buffer).strip()
        self._text_buffer.clear()

        if not full_text:
            return

        print(f"[Sarvam TTS] Synthesizing ({len(full_text)} chars): {full_text[:60]}...")
        await self._synthesize(full_text)

    async def _synthesize(self, text: str):
        """Use the REST API to synthesize the full utterance at once, ensuring high quality."""
        try:
            # Use REST API (batch) which is more reliable for bulbul:v3
            response = await _client.text_to_speech.convert(
                text=text,
                target_language_code=self.language,
                speaker=self.speaker,
                speech_sample_rate=8000,
                output_audio_codec="linear16",
                model=self.model,
                pace=1.1,
                enable_preprocessing=True
            )

            if not response.audios or len(response.audios) == 0:
                print("[Sarvam TTS] No audio returned from REST API.")
                return

            if self._cancelled:
                return

            # Decode the base64 raw PCM (linear16)
            raw_pcm = base64.b64decode(response.audios[0])
            
            # Convert to mulaw
            mulaw_bytes = audioop.lin2ulaw(raw_pcm, 2)
            
            # Stream the audio in small chunks to Twilio (e.g. 4000 bytes = 500ms at 8kHz)
            # This prevents Twilio from buffering a massive chunk and keeps latency smooth
            CHUNK_SIZE = 4000 
            for i in range(0, len(mulaw_bytes), CHUNK_SIZE):
                if self._cancelled:
                    break
                chunk = mulaw_bytes[i:min(i + CHUNK_SIZE, len(mulaw_bytes))]
                mulaw_b64 = base64.b64encode(chunk).decode("utf-8")
                await self.on_audio(mulaw_b64)
                # Sleep briefly to pace the sending (approx stream time)
                # Twilio expects 8000 bytes/sec natively
                await asyncio.sleep(len(chunk) / 8000.0 * 0.5)

        except Exception as e:
            if "1000" not in str(e) and "OK" not in str(e):
                print(f"\n[Sarvam TTS] Synthesis error: {e}")
        finally:
            self._active_ws = None

    async def clear_and_reconnect(self):
        """Barge-in: discard buffer and cancel active synthesis."""
        print("[Sarvam TTS] Barge-in: clearing")
        self._cancelled = True
        self._text_buffer.clear()

        # Force-close active WebSocket if any
        if self._active_ws:
            try:
                await self._active_ws.close()
            except Exception:
                pass

        # Ready for next utterance
        self._cancelled = False

    async def close(self):
        """Shutdown."""
        self._cancelled = True
        self._text_buffer.clear()
        if self._active_ws:
            try:
                await self._active_ws.close()
            except Exception:
                pass
        print("[Sarvam TTS] Disconnected")
