"""
Interactive chat with a trained TinyGPT model.

Usage:
    python interact.py
    python interact.py --checkpoint checkpoints/best.pt
    python interact.py --temperature 0.5 --top_k 10
    python interact.py --seed 42

Responses stream token-by-token. Type messages as the CLIENT. The model generates OUTPUT responses.
The conversation state accumulates across turns.
Commands: 'reset'/'clear' to clear, 'help' for help, 'quit' to exit.
"""

import argparse
import sys
import glob
import os
import torch
from model import TinyGPT
from config import ModelConfig
from vocabulary import (
    tokenize, detokenize, is_valid_sentence,
    SOS_ID, EOS_ID, OUTPUT_ID, CLIENT_ID, PAD_ID,
    ID_TO_WORD, WORD_TO_ID,
)

# Punctuation that attaches without leading space
_PUNCT_NO_SPACE = frozenset(".?,!")

# ANSI color codes
_CYAN = "\033[36m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_BLUE = "\033[34m"
_MAGENTA = "\033[35m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RESET = "\033[0m"

# Set in main() based on --color flag and TTY
_use_color = True

def _c(*codes: str) -> str:
    """Apply ANSI codes; no-op when colors disabled."""
    return "".join(codes) if _use_color else ""

def _s(text: str, *codes: str) -> str:
    """Style text with ANSI codes."""
    return _c(*codes) + text + _c(_RESET) if _use_color else text


def list_checkpoints(checkpoint_dir: str = "checkpoints"):
    """Find all .pt files in the checkpoint directory and return info about each."""
    pattern = os.path.join(checkpoint_dir, "*.pt")
    files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    checkpoints = []
    for path in files:
        try:
            meta = torch.load(path, map_location="cpu", weights_only=False)
            epoch = meta.get("epoch", "?")
            val_loss = meta.get("val_loss", "?")
            if isinstance(val_loss, float):
                val_loss = f"{val_loss:.4f}"
            checkpoints.append((path, epoch, val_loss))
        except Exception:
            checkpoints.append((path, "?", "?"))
    return checkpoints


def pick_checkpoint(checkpoint_dir: str = "checkpoints") -> str:
    """List available checkpoints and let the user pick one."""
    checkpoints = list_checkpoints(checkpoint_dir)
    if not checkpoints:
        print(_s(f"No checkpoints found in {checkpoint_dir}/", _YELLOW))
        raise SystemExit(1)

    print(_s("Available checkpoints", _BOLD, _CYAN))
    print()
    for i, (path, epoch, val_loss) in enumerate(checkpoints):
        name = os.path.basename(path)
        num = _s(f"[{i + 1}]", _BOLD, _BLUE)
        meta = _s(f"(epoch {epoch}, val_loss {val_loss})", _DIM)
        print(f"  {num} {name}  {meta}")
    print()

    while True:
        try:
            choice = input(_s("Select checkpoint ", _DIM) + _s(f"[1-{len(checkpoints)}]", _BLUE) + ": ").strip()
        except (EOFError, KeyboardInterrupt):
            raise SystemExit(0)
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(checkpoints):
                return checkpoints[idx][0]
        except ValueError:
            pass
        print(_s(f"  Please enter a number between 1 and {len(checkpoints)}", _YELLOW))


def load_model(checkpoint_path: str, device: torch.device):
    """Load a TinyGPT model from a training checkpoint.

    Checkpoints are dicts with keys: model_state_dict, config, epoch, val_loss.
    Returns (model, config) with the model in eval mode.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = TinyGPT(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    epoch = checkpoint.get("epoch", "?")
    val_loss = checkpoint.get("val_loss", "?")
    print(_s("Loaded ", _GREEN) + _s(os.path.basename(checkpoint_path), _BOLD) + _s(f" (epoch {epoch}, val_loss {val_loss})", _DIM))
    print(_s(f"Parameters: {model.count_parameters():,}", _DIM))
    return model, config


def build_conversation_tokens(turns, current_client_msg):
    """
    Build the full token sequence for the conversation so far,
    ending with OUTPUT: so the model generates the response.
    """
    parts = []

    # Previous turns
    for client_msg, output_msg in turns:
        parts.append(f"CLIENT:\n{client_msg}\n\nOUTPUT:\n{output_msg}")

    # Current turn: client message + OUTPUT: prefix
    parts.append(f"CLIENT:\n{current_client_msg}\n\nOUTPUT:")

    full_text = "\n\n".join(parts)
    ids = tokenize(full_text, add_special=True)
    # Remove <eos> at the end — we want the model to continue generating
    if ids and ids[-1] == EOS_ID:
        ids = ids[:-1]
    return ids


def generate_response(model, token_ids, config, device, temperature=0.8, top_k=20, max_tokens=40, on_token=None):
    """Generate model response tokens until CLIENT:, <eos>, or max_tokens.

    Unlike TinyGPT.generate(), this function returns only the newly generated
    token IDs (not the full sequence) and treats CLIENT: as a stop token so
    the model doesn't hallucinate the next user turn.

    If on_token(token_id) is provided, it is called with each new token as it
    is generated, enabling streaming output.
    """
    input_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)

    model.eval()
    generated = []
    with torch.no_grad():
        for _ in range(max_tokens):
            # Crop to max_seq_len
            ids = input_tensor if input_tensor.size(1) <= config.max_seq_len else input_tensor[:, -config.max_seq_len:]

            logits = model(ids)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = torch.nn.functional.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            token_id = next_token.item()

            # Stop on <eos> or CLIENT: (model tries to start a new turn)
            if token_id == EOS_ID or token_id == CLIENT_ID:
                break

            if on_token is not None:
                on_token(token_id)
            generated.append(token_id)
            input_tensor = torch.cat([input_tensor, next_token], dim=1)

    return generated


def main():
    parser = argparse.ArgumentParser(description="Chat with TinyGPT")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint (if not set, shows a picker)")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Directory to scan for checkpoints")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Sampling temperature (lower = more deterministic)")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Top-k sampling")
    parser.add_argument("--max_tokens", type=int, default=40,
                        help="Max tokens per response")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducible generation")
    parser.add_argument("--no-color", action="store_true", help="Disable colored output")
    args = parser.parse_args()

    global _use_color
    _use_color = not args.no_color and sys.stdout.isatty()

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    checkpoint_path = args.checkpoint or pick_checkpoint(args.checkpoint_dir)
    model, config = load_model(checkpoint_path, device)
    print(_s(f"Device: {device}", _DIM))
    print()
    print(_s("TinyGPT Interactive Chat", _BOLD, _MAGENTA))
    print(_s("=" * 42, _DIM))
    print(_s("Describe who has what, transfers, or ask questions.", _CYAN))
    print(_s("Commands: ", _DIM) + "reset/clear, help, quit")
    print(_s("=" * 42, _DIM))
    print()

    turns = []  # List of (client_msg, output_msg)

    while True:
        try:
            user_input = input(_s("You: ", _BOLD, _CYAN)).strip()
        except (EOFError, KeyboardInterrupt):
            print("\n" + _s("Bye!", _GREEN))
            break

        if not user_input:
            continue
        cmd = user_input.lower()
        if cmd == "quit":
            print(_s("Bye!", _GREEN))
            break
        if cmd in ("reset", "clear"):
            turns = []
            print(_s("[conversation cleared]", _DIM) + "\n")
            continue
        if cmd == "help":
            print()
            print(_s("Commands:", _BOLD, _CYAN))
            print("  reset, clear  — clear conversation history")
            print("  help          — show this message")
            print("  quit          — exit")
            print()
            print(_s("Example:", _BOLD, _CYAN))
            print(_s("  You:", _CYAN) + " Alice has the ball. Bob has the key.")
            print(_s("  You:", _CYAN) + " Alice gives the ball to Bob.")
            print(_s("  You:", _CYAN) + " Who has the ball?")
            print()
            continue

        valid, unknown = is_valid_sentence(user_input)
        if not valid:
            print(_s(f"Unknown words (will map to <unk>): {', '.join(unknown)}", _YELLOW))

        token_ids = build_conversation_tokens(turns, user_input)

        if args.seed is not None:
            torch.manual_seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed)

        first_token = [True]

        def stream_token(token_id):
            if token_id in (PAD_ID, SOS_ID, EOS_ID):
                return
            word = ID_TO_WORD.get(token_id, "<unk>")
            need_space = not first_token[0] and word not in _PUNCT_NO_SPACE
            print((" " if need_space else "") + word, end="", flush=True)
            first_token[0] = False
        print(_s("TinyGPT:", _BOLD, _GREEN) + " ", end="", flush=True)
        response_ids = generate_response(
            model, token_ids, config, device,
            temperature=args.temperature,
            top_k=args.top_k,
            max_tokens=args.max_tokens,
            on_token=stream_token,
        )
        response_text = detokenize(response_ids, strip_special=True)
        if not response_text.strip():
            print(_s("...", _DIM), end="")
        print()

        turns.append((user_input, response_text))


if __name__ == "__main__":
    main()
