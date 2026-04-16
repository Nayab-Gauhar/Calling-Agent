import gspread
from google.oauth2.service_account import Credentials
import asyncio
from datetime import datetime

# Path to the service account JSON
SERVICE_ACCOUNT_FILE = 'gen-lang-client-0724838448-59787ae79b2c.json'

# Google Sheets API scopes
SCOPES = [
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/drive'
]

# ID of the Google Sheet (extracted from the URL)
SPREADSHEET_ID = '1LXx0VyxuyKhkqUQRbEG4gQDDUOhyCYVha7BQDUd26jg'

def get_client():
    credentials = Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES
    )
    return gspread.authorize(credentials)

def _append_call_log(data: dict):
    try:
        client = get_client()
        sheet = client.open_by_key(SPREADSHEET_ID).sheet1
        
        # Prepare the row data
        row = [
            data.get("caller_number", "unknown"),
            data.get("direction", "unknown"),
            data.get("start_time", "").strftime("%Y-%m-%d %H:%M:%S") if isinstance(data.get("start_time"), datetime) else str(data.get("start_time", "")),
            data.get("end_time", "").strftime("%Y-%m-%d %H:%M:%S") if isinstance(data.get("end_time"), datetime) else str(data.get("end_time", "")),
            str(data.get("duration_seconds", 0)),
        ]
        
        # Append to the sheet
        sheet.append_row(row)
        print("[Google Sheets] Successfully appended call log")
    except Exception as e:
        print(f"[Google Sheets] Failed to append call log: {e}")

async def append_call_log(data: dict):
    # Run the synchronous gspread code in a thread pool to avoid blocking the asyncio event loop
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _append_call_log, data)
