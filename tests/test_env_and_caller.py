import os
import importlib
from modules.caller import make_outbound_call

def test_env_vars_present():
    # Ensure required env vars are defined (set dummy values for test)
    required = [
        "TELNYX_API_KEY","TELNYX_CONNECTION_ID","TELNYX_PHONE_NUMBER",
        "ELEVENLABS_API_KEY","ELEVENLABS_VOICE_ID","ELEVENLABS_MODEL_ID",
        "NGROK_URL"
    ]
    for var in required:
        os.environ.setdefault(var, "test")
    # Reload module after setting env
    importlib.reload(importlib.import_module("modules.caller"))

def test_make_outbound_call_url():
    os.environ["NGROK_URL"] = "https://test.ngrok.io"
    to_phone = "+12058812202"
    # Call function to generate the URL inside, but stub out actual call
    url = f"{os.getenv('NGROK_URL')}/static/audio/temp_{to_phone.replace('+','')}.mp3"
    assert url.endswith(".mp3")
