"""
Streaming Google Gemini LLM wrapper using the new google.genai SDK.
Streams Gemini responses token-by-token and buffers them into small chunks
for efficient, low-latency TTS processing. Supports cancellation for barge-in.
"""

import asyncio
from google import genai
from config import GEMINI_API_KEY, GEMINI_MODEL, SYSTEM_PROMPT, LLM_MAX_TOKENS

# Initialize the Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)

# Characters that indicate a natural pause — good point to flush to TTS
FLUSH_CHARS = {'.', '!', '?', ':', ';', ','}
# Minimum chunk size before we consider flushing to TTS
MIN_CHUNK_SIZE = 8


class GeminiLLM:
    """Manages conversation history and streams responses from Google Gemini."""

    def __init__(self, model=None, max_tokens=None, temperature=0.7):
        self.model_name = model or GEMINI_MODEL
        self.max_tokens = max_tokens or LLM_MAX_TOKENS
        self.temperature = temperature
        self.conversation_history = []  # list of {"role": ..., "content": ...}
        self.system_prompt = SYSTEM_PROMPT

    def add_message(self, role: str, content: str):
        """Add a message to conversation history."""
        self.conversation_history.append({"role": role, "content": content})
        # Keep history manageable (last 20 messages)
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]

    def _build_contents(self):
        """Convert conversation history to Gemini's content format."""
        contents = []
        for msg in self.conversation_history:
            role = "user" if msg["role"] == "user" else "model"
            contents.append(genai.types.Content(
                role=role,
                parts=[genai.types.Part(text=msg["content"])],
            ))
        return contents

    async def stream_response(self, user_message: str, cancel_event: asyncio.Event = None):
        """
        Stream a response from Gemini for the given user message.
        Yields small text chunks as soon as a natural pause is detected,
        optimized for lowest time-to-first-audio.

        Args:
            user_message: The user's transcribed speech.
            cancel_event: If set, stops generation immediately (for barge-in).

        Yields:
            str: Text chunks (roughly clause/sentence-sized).
        """
        self.add_message("user", user_message)

        max_retries = 3
        for attempt in range(max_retries):
            try:
                contents = self._build_contents()

                config = genai.types.GenerateContentConfig(
                    system_instruction=self.system_prompt,
                    max_output_tokens=self.max_tokens,
                    temperature=self.temperature,
                )

                # Use async streaming
                response = await client.aio.models.generate_content_stream(
                    model=self.model_name,
                    contents=contents,
                    config=config,
                )

                full_response = ""
                buffer = ""

                async for chunk in response:
                    # Check for barge-in cancellation
                    if cancel_event and cancel_event.is_set():
                        print("[Gemini] Generation cancelled (barge-in)")
                        break

                    if chunk.text:
                        token = chunk.text
                        full_response += token
                        buffer += token

                        # Flush buffer at natural pause points for low latency
                        if (
                            len(buffer) >= MIN_CHUNK_SIZE
                            and buffer.rstrip()[-1:] in FLUSH_CHARS
                        ):
                            yield buffer
                            buffer = ""

                # Yield any remaining text in the buffer
                if buffer.strip():
                    yield buffer

                # Save the full response to history
                if full_response:
                    self.add_message("assistant", full_response)
                    print(f"[Gemini] Response ({len(full_response)} chars): {full_response[:80]}...")
                return  # Success — exit retry loop

            except Exception as e:
                error_str = str(e)
                if "429" in error_str and attempt < max_retries - 1:
                    # Extract retry delay from error if available
                    wait = 5 * (attempt + 1)
                    if "retryDelay" in error_str:
                        try:
                            import re
                            match = re.search(r'"retryDelay":\s*"(\d+)s"', error_str)
                            if match:
                                wait = int(match.group(1)) + 1
                        except Exception:
                            pass
                    print(f"[Gemini] Rate limited, retrying in {wait}s (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(wait)
                    continue
                else:
                    print(f"[Gemini] Error streaming response: {e}")
                    error_msg = "I'm sorry, I didn't catch that. Could you repeat?"
                    self.add_message("assistant", error_msg)
                    yield error_msg
                    return

    def get_history(self):
        """Return conversation history."""
        return self.conversation_history

    def set_history(self, messages: list):
        """Set conversation history from a list of message dicts."""
        self.conversation_history = messages

    def clear_history(self):
        """Reset conversation history."""
        self.conversation_history = []
