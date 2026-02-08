from fastapi import FastAPI
from pydantic import BaseModel
from uuid import uuid4


# 1. Initialise a simple memory store (could be a dict, DB, etc.)
memory_store = {}          # key -> list of prior examples

def get_session_id(session_id):
    """Return the list of stored examples for this session."""
    return memory_store.get(session_id, []) or str(uuid4())  # generate new session_id if not provided

def get_short_memory(session_id):
    """Persist a new example (hotpath or background)."""
    return memory_store.setdefault(session_id, [])

def add_to_memory(session_id, user_msg, assistant_reply):
    get_short_memory(session_id).append(
        {"user": user_msg, "assistant": assistant_reply}
    )