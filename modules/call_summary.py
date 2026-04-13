"""
AI-powered call summarization using NVIDIA NIM LLM.

After each call, generates a structured summary with:
- Purpose of the call
- Lead status (Hot/Warm/Cold)
- Key details (budget, config, location, etc.)
- Brief conversation summary
- Recommended follow-up action

Returns a dict that can be written to Google Sheets or stored in MongoDB.
"""

import json
from openai import AsyncOpenAI
from config import NVIDIA_API_KEY, NVIDIA_MODEL

# Use NVIDIA NIM endpoint for summarization
_client = AsyncOpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY,
)

SUMMARY_PROMPT = """You are a real estate CRM assistant. Analyze the following phone call transcript between an AI sales agent (Aanya) and a potential buyer. Extract the following information and return ONLY a valid JSON object — no markdown, no explanation.

JSON structure:
{
  "caller_name": "Name if mentioned, otherwise 'Unknown'",
  "purpose": "Brief purpose of the call (e.g., '3BHK inquiry in Noida', 'Investment query', 'Follow-up on Trump Tower')",
  "lead_status": "Hot / Warm / Cold",
  "summary": "2-3 sentence summary of the conversation",
  "budget": "Budget range if mentioned, otherwise 'Not discussed'",
  "configuration": "BHK/villa/plot preference if mentioned, otherwise 'Not discussed'",
  "location_preference": "Preferred areas if mentioned, otherwise 'Not discussed'",
  "purpose_type": "End Use / Investment / Not discussed",
  "timeline": "When they want to buy if mentioned, otherwise 'Not discussed'",
  "follow_up": "Recommended next action (e.g., 'Send brochure', 'Schedule site visit', 'Call back in 2 days', 'No follow-up needed')"
}

Lead status rules:
- Hot: Expressed clear interest, shared budget/preferences, asked for details
- Warm: Showed some interest but didn't commit, asked general questions
- Cold: Not interested, hung up quickly, wrong number, asked not to call again

Transcript:
"""


async def generate_call_summary(conversation: list[dict]) -> dict | None:
    """
    Generate a structured summary of a call using NVIDIA LLM.

    Args:
        conversation: List of {"role": "user"/"assistant", "content": "..."} dicts.

    Returns:
        Dict with summary fields, or None if summarization fails.
    """
    if not conversation or not NVIDIA_API_KEY:
        return None

    # Build transcript text
    transcript_lines = []
    for msg in conversation:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            transcript_lines.append(f"Caller: {content}")
        elif role == "assistant":
            transcript_lines.append(f"Aanya: {content}")

    transcript = "\n".join(transcript_lines)

    if not transcript.strip():
        return None

    try:
        response = await _client.chat.completions.create(
            model=NVIDIA_MODEL,
            messages=[
                {"role": "system", "content": SUMMARY_PROMPT},
                {"role": "user", "content": transcript},
            ],
            max_tokens=500,
            temperature=0.3,
        )

        result_text = response.choices[0].message.content.strip()

        # Strip markdown code fences if present
        if result_text.startswith("```"):
            result_text = result_text.split("\n", 1)[-1]
            if result_text.endswith("```"):
                result_text = result_text[:-3].strip()

        summary = json.loads(result_text)
        print(f"[Summary] Generated: {summary.get('purpose', 'N/A')} / {summary.get('lead_status', 'N/A')}")
        return summary

    except json.JSONDecodeError as e:
        print(f"[Summary] Failed to parse LLM response as JSON: {e}")
        print(f"[Summary] Raw response: {result_text[:200]}")
        return None
    except Exception as e:
        print(f"[Summary] Error generating summary: {e}")
        return None
