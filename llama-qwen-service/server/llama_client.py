from __future__ import annotations

import os
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv()

LLAMA_BASE_URL = os.getenv("LLAMA_BASE_URL", "http://localhost:8080").rstrip("/")
LLAMA_MODEL = os.getenv("LLAMA_MODEL", "qwen2.5-1.5b-instruct")


async def chat_completion(messages: list[dict[str, Any]]) -> str:
    url = f"{LLAMA_BASE_URL}/v1/chat/completions"
    payload = {
        "model": LLAMA_MODEL,
        "messages": messages,
        "temperature": 0.2,
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
    return data["choices"][0]["message"]["content"]
