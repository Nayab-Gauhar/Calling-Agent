"""
MongoDB operations with async support via motor.
Uses lazy initialization so the app can start without a MongoDB URI.
"""

from motor.motor_asyncio import AsyncIOMotorClient
from config import MONGODB_URI

# Lazy MongoDB connection — only connect when URI is configured
_client = None
_db = None


def _get_db():
    """Get the database instance, initializing the connection if needed."""
    global _client, _db
    if _db is None:
        if not MONGODB_URI:
            return None
        _client = AsyncIOMotorClient(MONGODB_URI)
        _db = _client.Cluster0
    return _db


def _get_collection(name: str):
    """Get a collection by name, or None if DB is not configured."""
    db = _get_db()
    if db is None:
        return None
    return db[name]


# ─── User Operations ──────────────────────────────────────────────

async def get_user_by_phone(phone: str):
    """Find a user by phone number."""
    col = _get_collection("users")
    if col is None:
        return None
    return await col.find_one({"phone": phone})


async def get_userid_by_phone(phone: str):
    """Get user ID from phone number."""
    col = _get_collection("users")
    if col is None:
        return None
    user = await col.find_one({"phone": phone})
    if user:
        return user["_id"]
    return None


async def add_user(user_data: dict):
    """Create a new user."""
    col = _get_collection("users")
    if col is None:
        return None
    result = await col.insert_one(user_data)
    return result.inserted_id


async def has_interacted_before(phone: str) -> bool:
    """Check if a user has had a previous interaction."""
    col = _get_collection("users")
    if col is None:
        return False
    user = await col.find_one({"phone": phone})
    if user:
        return user.get("has_interacted_before", False)
    return False


async def set_interacted_before(phone: str):
    """Mark a user as having interacted."""
    col = _get_collection("users")
    if col is None:
        return
    await col.update_one(
        {"phone": phone},
        {"$set": {"has_interacted_before": True}},
        upsert=True,
    )


# ─── Chat History Operations ─────────────────────────────────────

async def get_chat_history(user_phone: str) -> list:
    """Retrieve chat history for a user by phone number."""
    col = _get_collection("chat_history")
    if col is None:
        return []
    record = await col.find_one({"phone": user_phone})
    if record and "messages" in record:
        return record["messages"]
    return []


async def save_chat_history(user_phone: str, messages: list):
    """Save or update the full chat history for a user."""
    col = _get_collection("chat_history")
    if col is None:
        return
    await col.update_one(
        {"phone": user_phone},
        {"$set": {"messages": messages}},
        upsert=True,
    )


async def append_to_chat_history(user_phone: str, role: str, content: str):
    """Append a single message to a user's chat history."""
    col = _get_collection("chat_history")
    if col is None:
        return
    await col.update_one(
        {"phone": user_phone},
        {"$push": {"messages": {"role": role, "content": content}}},
        upsert=True,
    )


# ─── Call Log Operations ─────────────────────────────────────────

async def save_call_log(call_data: dict):
    """Save metadata about a call (duration, direction, etc.)."""
    col = _get_collection("call_logs")
    if col is None:
        return None
    result = await col.insert_one(call_data)
    return result.inserted_id


# ─── Appointment Operations ──────────────────────────────────────

async def book_appointment(user_phone: str, appointment_data: dict):
    """Book a new appointment."""
    col = _get_collection("appointments")
    if col is None:
        return None
    appointment_data["phone"] = user_phone
    result = await col.insert_one(appointment_data)
    return result.inserted_id


async def get_appointments(user_phone: str):
    """Get all appointments for a user."""
    col = _get_collection("appointments")
    if col is None:
        return []
    cursor = col.find({"phone": user_phone})
    return await cursor.to_list(length=100)