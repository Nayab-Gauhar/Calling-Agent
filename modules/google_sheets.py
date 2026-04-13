"""
Google Sheets integration for call logging and chat history.

Tab 1 ("Call Logs"): One row per call with AI-generated summary.
Tab 2 ("Chat History"): Full conversation history per phone number.
    Enables Aanya to "remember" repeat callers across calls.

Uses gspread with a service account for server-to-server auth.
Silently no-ops if not configured (no creds file or sheet ID).

Setup:
  1. Enable Google Sheets API in Google Cloud Console
  2. Create a service account and download the JSON key
  3. Share your Google Sheet with the service account email
  4. Set GOOGLE_SHEETS_CREDS_FILE and GOOGLE_SHEET_ID in .env
"""

import asyncio
from datetime import datetime, timezone
from pathlib import Path

from config import GOOGLE_SHEETS_CREDS_FILE, GOOGLE_SHEET_ID

# Lazy-loaded gspread worksheets
_call_log_sheet = None
_chat_history_sheet = None
_initialized = False


def _init_sheets():
    """Lazy-init both worksheet tabs. Call once."""
    global _call_log_sheet, _chat_history_sheet, _initialized

    if _initialized:
        return

    _initialized = True

    if not GOOGLE_SHEETS_CREDS_FILE or not GOOGLE_SHEET_ID:
        print("[Google Sheets] Not configured — skipping")
        return

    creds_path = Path(GOOGLE_SHEETS_CREDS_FILE)
    if not creds_path.exists():
        print(f"[Google Sheets] Credentials file not found: {creds_path}")
        return

    try:
        import gspread
        gc = gspread.service_account(filename=str(creds_path))
        spreadsheet = gc.open_by_key(GOOGLE_SHEET_ID)

        # Tab 1: Call Logs (first sheet)
        _call_log_sheet = spreadsheet.sheet1
        _call_log_sheet.update_title("Call Logs")

        # Tab 2: Chat History (create if missing)
        try:
            _chat_history_sheet = spreadsheet.worksheet("Chat History")
        except gspread.exceptions.WorksheetNotFound:
            _chat_history_sheet = spreadsheet.add_worksheet(
                title="Chat History", rows=1000, cols=4
            )
            _chat_history_sheet.append_row(
                ["Phone", "Role", "Content", "Timestamp"]
            )
            print("[Google Sheets] Created 'Chat History' tab")

        print(f"[Google Sheets] Connected: {spreadsheet.title}")
    except Exception as e:
        print(f"[Google Sheets] Init error: {e}")


def _get_call_log_sheet():
    _init_sheets()
    return _call_log_sheet


def _get_chat_history_sheet():
    _init_sheets()
    return _chat_history_sheet


def is_configured() -> bool:
    """Check if Google Sheets logging is enabled."""
    return bool(GOOGLE_SHEETS_CREDS_FILE and GOOGLE_SHEET_ID)


# ─── Call Logs (Tab 1) ────────────────────────────────────────

async def ensure_headers():
    """Add header row to Call Logs if the sheet is empty."""
    sheet = _get_call_log_sheet()
    if sheet is None:
        return

    try:
        first_row = await asyncio.to_thread(sheet.row_values, 1)
        if first_row:
            return

        headers = [
            "Timestamp", "Phone Number", "Direction", "Duration",
            "Purpose", "Lead Status", "Summary", "Budget",
            "Configuration", "Location", "Purpose Type", "Timeline",
            "Follow-up", "Caller Name",
        ]
        await asyncio.to_thread(sheet.append_row, headers)
        print("[Google Sheets] Call Logs headers added")
    except Exception as e:
        print(f"[Google Sheets] Error adding headers: {e}")


async def log_call(
    caller_number: str,
    direction: str,
    duration_seconds: float,
    summary: dict | None = None,
):
    """Append a call log row."""
    sheet = _get_call_log_sheet()
    if sheet is None:
        return

    await ensure_headers()

    mins = int(duration_seconds) // 60
    secs = int(duration_seconds) % 60
    dur_str = f"{mins}m {secs}s" if mins else f"{secs}s"
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    s = summary or {}
    row = [
        timestamp, caller_number, direction, dur_str,
        s.get("purpose", "N/A"), s.get("lead_status", "N/A"),
        s.get("summary", "N/A"), s.get("budget", "Not discussed"),
        s.get("configuration", "Not discussed"),
        s.get("location_preference", "Not discussed"),
        s.get("purpose_type", "Not discussed"),
        s.get("timeline", "Not discussed"),
        s.get("follow_up", "N/A"), s.get("caller_name", "Unknown"),
    ]

    try:
        await asyncio.to_thread(sheet.append_row, row, value_input_option="USER_ENTERED")
        print(f"[Google Sheets] Call logged: {caller_number} / {s.get('lead_status', 'N/A')}")
    except Exception as e:
        print(f"[Google Sheets] Error logging call: {e}")


# ─── Chat History (Tab 2) ─────────────────────────────────────

async def get_chat_history(phone: str) -> list[dict]:
    """
    Load chat history for a phone number.
    Returns list of {"role": ..., "content": ...} dicts.
    """
    sheet = _get_chat_history_sheet()
    if sheet is None:
        return []

    try:
        all_rows = await asyncio.to_thread(sheet.get_all_values)
        # Skip header row, filter by phone number
        messages = []
        for row in all_rows[1:]:
            if len(row) >= 3 and row[0] == phone:
                messages.append({"role": row[1], "content": row[2]})
        return messages
    except Exception as e:
        print(f"[Google Sheets] Error loading history for {phone}: {e}")
        return []


async def append_to_chat_history(phone: str, role: str, content: str):
    """Append a single message to chat history."""
    sheet = _get_chat_history_sheet()
    if sheet is None:
        return

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    row = [phone, role, content, timestamp]

    try:
        await asyncio.to_thread(sheet.append_row, row)
    except Exception as e:
        print(f"[Google Sheets] Error appending history: {e}")


async def save_chat_history(phone: str, messages: list[dict]):
    """
    Save final conversation history. Since we already append in real-time,
    this is a no-op — kept for API compatibility.
    """
    pass
