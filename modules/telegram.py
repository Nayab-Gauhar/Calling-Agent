"""
Telegram Bot integration for call notifications.
Sends call summaries (caller, direction, duration, full transcript) to a
Telegram chat after each call ends. Uses the Telegram Bot HTTP API directly
via httpx — no extra dependencies required.

Silently no-ops if TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID is not configured.
"""

import httpx
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

TELEGRAM_API = "https://api.telegram.org"

# Reusable async client (connection pooling)
_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient | None:
    """Lazy-init the httpx client. Returns None if Telegram is not configured."""
    global _client
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return None
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(timeout=10.0)
    return _client


def is_configured() -> bool:
    """Check if Telegram notifications are enabled."""
    return bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)


async def send_message(text: str, parse_mode: str = "HTML") -> bool:
    """
    Send a message to the configured Telegram chat.

    Args:
        text: Message text (supports HTML formatting).
        parse_mode: "HTML" or "MarkdownV2".

    Returns:
        True if sent successfully, False otherwise.
    """
    client = _get_client()
    if client is None:
        return False

    url = f"{TELEGRAM_API}/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": parse_mode,
    }

    try:
        resp = await client.post(url, json=payload)
        if resp.status_code == 200:
            return True
        print(f"[Telegram] Send failed: {resp.status_code} — {resp.text[:200]}")
        return False
    except Exception as e:
        print(f"[Telegram] Error sending message: {e}")
        return False


def format_call_summary(
    caller_number: str,
    direction: str,
    stream_sid: str | None,
    duration_seconds: float,
    conversation: list[dict],
) -> str:
    """
    Build a nicely formatted HTML message for Telegram.

    Args:
        caller_number: Phone number of the caller.
        direction: "inbound" or "outbound".
        stream_sid: Twilio stream SID.
        duration_seconds: Call duration in seconds.
        conversation: List of {"role": "user"/"assistant", "content": "..."} dicts.

    Returns:
        HTML-formatted string ready to send via Telegram.
    """
    # Duration formatting
    mins = int(duration_seconds) // 60
    secs = int(duration_seconds) % 60
    dur_str = f"{mins}m {secs}s" if mins else f"{secs}s"

    # Direction emoji
    dir_icon = "\u2b05\ufe0f" if direction == "inbound" else "\u27a1\ufe0f"  # arrows

    # Header
    lines = [
        f"<b>{dir_icon} {'Incoming' if direction == 'inbound' else 'Outgoing'} Call</b>",
        "",
        f"<b>Phone:</b> <code>{_escape_html(caller_number)}</code>",
        f"<b>Duration:</b> {dur_str}",
    ]
    if stream_sid:
        lines.append(f"<b>Stream SID:</b> <code>{_escape_html(stream_sid)}</code>")

    # Transcript
    if conversation:
        lines.append("")
        lines.append("<b>Conversation:</b>")
        lines.append("")
        for msg in conversation:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                lines.append(f"\U0001f464 <b>User:</b> {_escape_html(content)}")
            elif role == "assistant":
                lines.append(f"\U0001f916 <b>AI:</b> {_escape_html(content)}")
        lines.append("")
    else:
        lines.append("")
        lines.append("<i>No conversation recorded.</i>")

    return "\n".join(lines)


async def send_call_summary(
    caller_number: str,
    direction: str,
    stream_sid: str | None,
    duration_seconds: float,
    conversation: list[dict],
) -> bool:
    """
    Format and send a call summary to Telegram. Convenience wrapper.

    Returns True if sent, False if not configured or failed.
    """
    if not is_configured():
        return False

    text = format_call_summary(
        caller_number, direction, stream_sid, duration_seconds, conversation
    )

    # Telegram has a 4096 char limit per message.
    # If the transcript is too long, split it.
    if len(text) <= 4096:
        return await send_message(text)

    # Split: send header first, then transcript in chunks
    header_end = text.find("<b>Conversation:</b>")
    if header_end == -1:
        # No conversation section — just truncate
        return await send_message(text[:4090] + "...")

    header = text[:header_end].rstrip()
    transcript = text[header_end:]

    await send_message(header)

    # Send transcript in 4000-char chunks
    while transcript:
        chunk = transcript[:4000]
        transcript = transcript[4000:]
        if transcript:
            chunk += "..."
        await send_message(chunk)

    return True


def _escape_html(text: str) -> str:
    """Escape HTML special characters for Telegram."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
