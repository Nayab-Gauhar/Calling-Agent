"""
Streaming LLM wrapper.
Uses OpenAI SDK pointing to Groq API to stream Llama-3 responses token-by-token
and buffers them into small chunks for efficient, low-latency TTS processing.
Supports cancellation for barge-in.
"""

import asyncio
from openai import AsyncOpenAI
from config import GROQ_API_KEY, GROQ_MODEL, SYSTEM_PROMPT, LLM_MAX_TOKENS

# Initialize the OpenAI client pointing to Groq API
client = AsyncOpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY
)

# Characters that indicate a natural pause — good point to flush to TTS.
FLUSH_CHARS = {'.', '!', '?', ':', ';', ',', '।', '॥', '—', '–'}
MIN_CHUNK_SIZE = 12


class GroqLLM:
    """Manages conversation history and streams responses from Groq API."""

    def __init__(self, model=None, max_tokens=None, temperature=0.7):
        self.model_name = model or GROQ_MODEL
        self.max_tokens = max_tokens or LLM_MAX_TOKENS
        self.temperature = temperature
        self.conversation_history = []  # list of {"role": ..., "content": ...}
        self.system_prompt = SYSTEM_PROMPT

    def add_message(self, role: str, content: str):
        """Add a message to conversation history."""
        self.conversation_history.append({"role": role, "content": content})
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]

    def _build_messages(self):
        """Convert conversation history to message format."""
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.conversation_history)
        return messages

    async def stream_response(self, user_message: str, cancel_event: asyncio.Event = None):
        """Stream a response for the given user message."""
        self.add_message("user", user_message)

        try:
            messages = self._build_messages()

            response = await client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stream=True,
            )

            full_response = ""
            buffer = ""
            was_cancelled = False

            async for chunk in response:
                if cancel_event and cancel_event.is_set():
                    print("[Groq] Generation cancelled (barge-in)")
                    was_cancelled = True
                    break

                if chunk.choices and chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    full_response += token
                    buffer += token

                    if (
                        len(buffer) >= MIN_CHUNK_SIZE
                        and buffer.rstrip()[-1:] in FLUSH_CHARS
                    ):
                        yield buffer
                        buffer = ""
                    elif len(buffer) >= 80:
                        yield buffer
                        buffer = ""

            if buffer.strip() and not was_cancelled:
                yield buffer

            if full_response and not was_cancelled:
                self.add_message("assistant", full_response)
                print(f"[Groq] Response ({len(full_response)} chars): {full_response[:80]}...")
            elif was_cancelled and full_response:
                if self.conversation_history and self.conversation_history[-1].get("role") == "user":
                    self.conversation_history.pop()
                print(f"[Groq] Discarded partial response ({len(full_response)} chars) due to barge-in")

        except Exception as e:
            print(f"[Groq] Error streaming response: {e}")
            error_msg = "Sorry, I didn't quite catch that. Could you say that again?"
            self.add_message("assistant", error_msg)
            yield error_msg

    def get_history(self):
        return self.conversation_history

    def set_history(self, messages: list):
        self.conversation_history = messages

    def clear_history(self):
        self.conversation_history = []

    async def stream_greeting(self, cancel_event: asyncio.Event = None):
        """Generate an opening greeting for the call."""
        try:
            messages = [{"role": "system", "content": self.system_prompt}]
            messages.extend(self.conversation_history)
            messages.append({
                "role": "user",
                "content": (
                    "The phone call just connected. Deliver your opening line naturally. "
                    "Since you don't know the caller's name yet, skip the [Name] placeholder "
                    "and go straight to introducing yourself and asking if it's a good time. "
                    "Keep it under 2 sentences."
                ),
            })

            response = await client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=120,
                temperature=self.temperature,
                stream=True,
            )

            full_response = ""
            buffer = ""

            async for chunk in response:
                if cancel_event and cancel_event.is_set():
                    break

                if chunk.choices and chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    full_response += token
                    buffer += token

                    if (
                        len(buffer) >= MIN_CHUNK_SIZE
                        and buffer.rstrip()[-1:] in FLUSH_CHARS
                    ):
                        yield buffer
                        buffer = ""
                    elif len(buffer) >= 80:
                        yield buffer
                        buffer = ""

            if buffer.strip():
                yield buffer

            if full_response:
                self.add_message("assistant", full_response)
                print(f"[Groq] Greeting ({len(full_response)} chars): {full_response[:80]}...")

        except Exception as e:
            print(f"[Groq] Error generating greeting: {e}")
            fallback = "Hi, this is Aanya calling from Trump Tower Noida. Am I speaking with the right person?"
            self.add_message("assistant", fallback)
            yield fallback
