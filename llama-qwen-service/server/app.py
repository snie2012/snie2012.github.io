from __future__ import annotations

import os
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from llama_client import chat_completion
from tooling import TOOL_SCHEMA, parse_tool_call

load_dotenv()

MAX_TOOL_CALLS = int(os.getenv("MAX_TOOL_CALLS", "1"))
USER_AGENT = os.getenv("USER_AGENT", "LlamaQwenService/0.1")

app = FastAPI()


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str


def build_system_prompt() -> str:
    return (
        "You are a helpful assistant. You can optionally call a tool by responding "
        "with pure JSON in the format: {\"tool\": \"fetch_url\", \"args\": {\"url\": \"https://...\"}}. "
        "Only respond with the JSON object when you want to call the tool. "
        f"Tool schema: {TOOL_SCHEMA}"
    )


async def fetch_url(url: str) -> str:
    headers = {"User-Agent": USER_AGENT}
    async with httpx.AsyncClient(timeout=30.0, headers=headers) as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.text[:4000]


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": request.message},
    ]

    tool_calls = 0
    while True:
        reply = await chat_completion(messages)
        tool_call = parse_tool_call(reply)
        if tool_call is None:
            return ChatResponse(reply=reply)
        if tool_calls >= MAX_TOOL_CALLS:
            return ChatResponse(reply="Tool call limit reached.")
        tool_calls += 1
        if tool_call.name != "fetch_url":
            return ChatResponse(reply="Unsupported tool call.")
        try:
            result = await fetch_url(tool_call.args["url"])
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "user", "content": f"Tool result: {result}"})
