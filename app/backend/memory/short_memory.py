from uuid import uuid4


# 1. Initialise a simple memory store (could be a dict, DB, etc.)
memory_store = {}          # key -> list of prior examples

def get_session_id(session_id):
    """Return a valid session_id, generating new one if not provided."""
    if session_id and session_id in memory_store:
        return session_id
    new_id = str(uuid4())
    memory_store[new_id] = []
    return new_id

def get_short_memory(session_id):
    """Get memory list for the session."""
    return memory_store.setdefault(session_id, [])

def add_to_memory(session_id, user_msg, assistant_reply):
    get_short_memory(session_id).append(
        {"user": user_msg, "assistant": assistant_reply}
    )