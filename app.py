"""
FastAPI web UI for TinyGPT.

Usage:
    python app.py
    python app.py --checkpoint checkpoints/best.pt
    python app.py --port 8000 --host 0.0.0.0

Opens a browser chat interface at http://localhost:8000
"""

from __future__ import annotations
import argparse
import asyncio
import os
import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from interact import (
    load_model,
    build_conversation_tokens,
    generate_response,
    list_checkpoints,
)
from vocabulary import detokenize, ID_TO_WORD, PAD_ID, SOS_ID, EOS_ID

_PUNCT_NO_SPACE = frozenset(".?,!")
DEFAULT_CHECKPOINT = "checkpoints/best.pt"

app = FastAPI()

# Global model state
_model = None
_config = None
_device = None
_checkpoint_path = None


class ChatRequest(BaseModel):
    messages: list[dict]  # [{role: "user"|"assistant", content: str}]
    temperature: float = 0.1
    top_k: int = 5


def _messages_to_turns(messages: list[dict]) -> tuple[list[tuple], str]:
    """Convert [{role, content}] to (turns, current_msg).

    turns is a list of (client_msg, output_msg) for completed exchanges.
    current_msg is the last user message (not yet responded to).
    """
    turns = []
    i = 0
    while i < len(messages) - 1:
        if messages[i]["role"] == "user" and messages[i + 1]["role"] == "assistant":
            turns.append((messages[i]["content"], messages[i + 1]["content"]))
            i += 2
        else:
            i += 1
    # Last message must be the current user message
    current_msg = messages[-1]["content"] if messages and messages[-1]["role"] == "user" else ""
    return turns, current_msg


def _token_to_word(token_id: int):
    """Convert a token ID to a displayable word. Returns None for special tokens."""
    if token_id in (PAD_ID, SOS_ID, EOS_ID):
        return None
    return ID_TO_WORD.get(token_id, "<unk>")


@app.get("/", response_class=HTMLResponse)
async def index():
    with open(os.path.join(os.path.dirname(__file__), "static", "index.html")) as f:
        return f.read()


@app.post("/chat")
async def chat(req: ChatRequest):
    if not req.messages or req.messages[-1]["role"] != "user":
        return JSONResponse({"error": "Last message must be from user"}, status_code=400)

    turns, current_msg = _messages_to_turns(req.messages)
    if not current_msg.strip():
        return JSONResponse({"error": "Empty message"}, status_code=400)

    token_ids = build_conversation_tokens(turns, current_msg)

    # Use a queue to bridge the sync generate_response callback with async SSE
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def on_token(token_id: int):
        word = _token_to_word(token_id)
        if word is not None:
            loop.call_soon_threadsafe(queue.put_nowait, word)

    async def run_generation():
        await asyncio.to_thread(
            generate_response,
            _model, token_ids, _config, _device,
            req.temperature, req.top_k, 40,
            on_token,
        )
        loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel

    async def event_stream():
        task = asyncio.create_task(run_generation())
        first = True
        try:
            while True:
                word = await queue.get()
                if word is None:
                    break
                # Prepend a space unless it's the first token or punctuation
                prefix = "" if first or word in _PUNCT_NO_SPACE else " "
                yield f"data: {prefix}{word}\n\n"
                first = False
        finally:
            task.cancel()
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/info")
async def info():
    if _model is None:
        return JSONResponse({"error": "Model not loaded"}, status_code=503)
    checkpoint = torch.load(_checkpoint_path, map_location="cpu", weights_only=False)
    epoch = checkpoint.get("epoch", "?")
    val_loss = checkpoint.get("val_loss", "?")
    if isinstance(val_loss, float):
        val_loss = round(val_loss, 4)
    return {
        "checkpoint": os.path.basename(_checkpoint_path),
        "epoch": epoch,
        "val_loss": val_loss,
        "parameters": _model.count_parameters(),
        "device": str(_device),
    }


def main():
    global _model, _config, _device, _checkpoint_path

    parser = argparse.ArgumentParser(description="TinyGPT FastAPI Web UI")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            _device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            _device = torch.device("mps")
        else:
            _device = torch.device("cpu")
    else:
        _device = torch.device(args.device)

    checkpoint_path = args.checkpoint or DEFAULT_CHECKPOINT
    if not os.path.isfile(checkpoint_path):
        checkpoints = list_checkpoints(args.checkpoint_dir)
        if checkpoints:
            checkpoint_path = checkpoints[0][0]
        else:
            raise SystemExit(
                "No checkpoint found. Train a model first:\n"
                "  python data_generator.py --train 300000 --val 2000 --outdir data\n"
                "  python train.py\n"
                "Then run: python app.py"
            )

    _checkpoint_path = checkpoint_path
    _model, _config = load_model(checkpoint_path, _device)
    print(f"Loaded {checkpoint_path} on {_device}")
    print(f"Open http://{args.host}:{args.port} in your browser")

    # Mount static files after startup
    app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
