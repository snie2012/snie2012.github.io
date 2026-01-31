# Llama.cpp Qwen Internet Tooling Service

This folder is a **new, minimal repo** for running Qwen models with `llama.cpp`, adding a tiny tool-calling loop to fetch URLs, and exposing the service over HTTP so you can use it from your phone or share it with friends.

## Goals (incremental)
1. **Run Qwen with llama.cpp** (local inference)
2. **Add a single tool** to fetch URLs from the internet
3. **Expose the service** over HTTP so it can be accessed remotely

## Prerequisites
- Docker (recommended) or a local build of `llama.cpp`
- Python 3.10+

## Step 1 — Run llama.cpp server with a Qwen model

Download/convert a Qwen GGUF model (example: `Qwen2.5-1.5B-Instruct`). Then run `llama.cpp`'s server.

**Docker (recommended):**
```bash
cd llama-qwen-service
mkdir -p models
# Place your GGUF model in ./models (example: qwen2.5-1.5b-instruct.gguf)

docker compose up llama
```

The llama.cpp server will start on `http://localhost:8080`.

## Step 2 — Run the simple tool-calling API

```bash
cd llama-qwen-service/server
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app:app --host 0.0.0.0 --port 8000
```

The service listens on `http://localhost:8000`.

## Step 3 — Call it from your phone / share with friends

You can expose port `8000` using a tunnel:

**Cloudflare Tunnel (free):**
```bash
cloudflared tunnel --url http://localhost:8000
```

**Tailscale Funnel** or **ngrok** also works. Share the public URL and call:

```bash
curl -X POST "$PUBLIC_URL/chat" \
  -H "Content-Type: application/json" \
  -d '{"message":"Fetch https://example.com and summarize."}'
```

## API usage

### POST `/chat`
Request:
```json
{ "message": "What is on https://example.com?" }
```

Response:
```json
{ "reply": "..." }
```

## Notes
- The tool-calling loop is intentionally simple: the model may emit a JSON tool call for `fetch_url`.
- See `server/tooling.py` for the tool schema and parser.
