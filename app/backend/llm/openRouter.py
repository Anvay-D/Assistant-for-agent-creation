# llm_openrouter.py
import requests
import os
from config import OPENROUTER_MODEL , OPENROUTER_API_KEY, OPENROUTER_URL

OPENROUTER_API_KEY = OPENROUTER_API_KEY
MODEL = OPENROUTER_MODEL 

def call_llm(prompt: str) -> str:
    response = requests.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": "You are a precise technical assistant who specializes in API documentation such as Qdrant and LangChain. Answer only from provided context with api and curl calls of the api in response. Do not respond with api that are marked as deprecated"},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        },
        timeout=60,
    )

    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]
