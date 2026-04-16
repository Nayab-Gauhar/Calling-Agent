import gspread
from google.oauth2.service_account import Credentials
import asyncio
from datetime import datetime
import json
from groq import AsyncGroq
from config import GROQ_API_KEY, GROQ_MODEL

# Path to the service account JSON
SERVICE_ACCOUNT_FILE = 'gen-lang-client-0724838448-59787ae79b2c.json'

# Google Sheets API scopes
SCOPES = [
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/drive'
]

# ID of the Google Sheet (extracted from the URL)
SPREADSHEET_ID = '1LXx0VyxuyKhkqUQRbEG4gQDDUOhyCYVha7BQDUd26jg'

# Initialize the Groq client for data extraction
groq_client = AsyncGroq(api_key=GROQ_API_KEY)

def get_client():
    credentials = Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES
    )
    return gspread.authorize(credentials)

async def extract_health_data_from_transcript(conversation: list) -> dict:
    """
    Uses Groq LLM to extract key healthcare info from the conversation history.
    """
    if not conversation:
        return {}

    # Format transcript for the prompt
    transcript = "\\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in conversation])
    
    prompt = f"""
    Analyze this healthcare voice assistant conversation transcript. 
    Extract the following details and return ONLY a valid JSON object.
    If a detail is not mentioned, use "Not mentioned" or null.
    
    Transcript:
    {transcript}
    
    Expected JSON Format:
    {{
        "patient_name": "String or Not mentioned",
        "symptoms": "Brief summary of symptoms",
        "suggested_action_or_doctor": "Doctor specialization or action suggested (e.g., General Physician, Visit ER, Rest)",
        "location": "City or location mentioned",
        "appointment_details": "Time/Date or Not mentioned",
        "language_used": "Hindi, English, or Hinglish"
    }}
    """
    
    try:
        response = await groq_client.chat.completions.create(
            model="llama3-8b-8192", # Using a fast model for extraction
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        print(f"[Google Sheets] Failed to extract data using LLM: {e}")
        return {}

def _append_call_log(data: dict):
    try:
        client = get_client()
        sheet = client.open_by_key(SPREADSHEET_ID).sheet1
        
        # We ensure headers exist (optional but good for hackathon)
        try:
            # Check if empty
            if not sheet.row_values(1):
                headers = [
                    "Timestamp", "Caller Number", "Direction", "Duration (s)", 
                    "Patient Name", "Symptoms / Problem", "Suggested Action / Doctor", 
                    "Location / City", "Appointment Details", "Language"
                ]
                sheet.append_row(headers)
        except Exception:
            pass # Ignore header creation errors if it already has data
            
        start_time_str = data.get("start_time", "").strftime("%Y-%m-%d %H:%M:%S") if isinstance(data.get("start_time"), datetime) else str(data.get("start_time", ""))
            
        extracted = data.get("extracted_data", {})
        
        # Prepare the row data based on the hackathon requirements
        row = [
            start_time_str,
            data.get("caller_number", "unknown"),
            data.get("direction", "unknown"),
            str(data.get("duration_seconds", 0)),
            extracted.get("patient_name", "Not mentioned"),
            extracted.get("symptoms", "Not mentioned"),
            extracted.get("suggested_action_or_doctor", "Not mentioned"),
            extracted.get("location", "Not mentioned"),
            extracted.get("appointment_details", "Not mentioned"),
            extracted.get("language_used", "Not mentioned"),
        ]
        
        # Append to the sheet
        sheet.append_row(row)
        print("[Google Sheets] Successfully appended detailed health call log")
    except Exception as e:
        print(f"[Google Sheets] Failed to append call log: {e}")

async def append_call_log(data: dict, conversation: list = None):
    # 1. Extract data asynchronously using Groq LLM
    if conversation:
        extracted_data = await extract_health_data_from_transcript(conversation)
        data["extracted_data"] = extracted_data
    else:
        data["extracted_data"] = {}
        
    # 2. Run the synchronous gspread code in a thread pool
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _append_call_log, data)
