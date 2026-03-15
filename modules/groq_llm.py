"""
Streaming Groq LLM wrapper.
Streams Llama-3 (or other Groq models) responses token-by-token and buffers them into small chunks
for efficient, low-latency TTS processing. Supports cancellation for barge-in.
"""

import asyncio
from groq import AsyncGroq
from config import GROQ_API_KEY, GROQ_MODEL, SYSTEM_PROMPT, LLM_MAX_TOKENS

# Initialize the Groq client
client = AsyncGroq(api_key=GROQ_API_KEY)

# Characters that indicate a natural pause — good point to flush to TTS.
# Includes Hindi danda (।) and double-danda (॥) which are sentence terminators,
# and em-dash / en-dash which LLMs frequently produce.
FLUSH_CHARS = {'.', '!', '?', ':', ';', ',', '।', '॥', '—', '–'}
# Minimum chunk size (in chars) before we consider flushing to TTS.
# Lower = faster time-to-first-audio, but too-short chunks sound choppy.
MIN_CHUNK_SIZE = 6


class GroqLLM:
    """Manages conversation history and streams responses from Groq."""

    def __init__(self, model=None, max_tokens=None, temperature=0.85):
        self.model_name = model or GROQ_MODEL
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

    def _build_messages(self):
        """Convert conversation history to Groq's message format."""
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.conversation_history)
        return messages

    async def stream_response(self, user_message: str, cancel_event: asyncio.Event = None):
        """
        Stream a response from Groq for the given user message.
        Yields small text chunks as soon as a natural pause is detected,
        optimized for lowest time-to-first-audio.

        Args:
            user_message: The user's transcribed speech.
            cancel_event: If set, stops generation immediately (for barge-in).

        Yields:
            str: Text chunks (roughly clause/sentence-sized).
        """
        self.add_message("user", user_message)

        try:
            messages = self._build_messages()

            # Use async streaming
            response = await client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stream=True,
            )

            full_response = ""
            buffer = ""

            async for chunk in response:
                # Check for barge-in cancellation
                if cancel_event and cancel_event.is_set():
                    print("[Groq] Generation cancelled (barge-in)")
                    break

                if chunk.choices and chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
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
                print(f"[Groq] Response ({len(full_response)} chars): {full_response[:80]}...")

        except Exception as e:
            print(f"[Groq] Error streaming response: {e}")
            error_msg = "I'm sorry, I didn't catch that. Could you repeat?"
            self.add_message("assistant", error_msg)
            yield error_msg

    def get_history(self):
        """Return conversation history."""
        return self.conversation_history

    def set_history(self, messages: list):
        """Set conversation history from a list of message dicts."""
        self.conversation_history = messages

    def clear_history(self):
        """Reset conversation history."""
        self.conversation_history = []
