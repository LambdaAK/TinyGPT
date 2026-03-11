"""
Training script for WorldLLM.

Usage:
    python train.py
    python train.py --epochs 30 --batch_size 32 --lr 1e-4
"""

import argparse
import os
import time
import torch
import torch.nn as nn
from config import ModelConfig, TrainConfig
from dataset import create_dataloader
from model import WorldLLM
from vocabulary import PAD_ID, VOCAB_SIZE


def get_device(preference: str) -> torch.device:
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(preference)


def masked_loss(logits, targets, loss_mask):
    """Compute cross-entropy only on positions where loss_mask == 1."""
    per_token_loss = nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
        ignore_index=PAD_ID,
        reduction="none",
    )
    flat_mask = loss_mask.view(-1)
    masked = per_token_loss * flat_mask
    num_tokens = flat_mask.sum()
    if num_tokens == 0:
        return masked.sum(), 0
    return masked.sum(), num_tokens.item()


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for inputs, targets, loss_mask in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            loss_mask = loss_mask.to(device)
            logits = model(inputs)
            loss, n = masked_loss(logits, targets, loss_mask)
            total_loss += loss.item()
            total_tokens += n

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = min(torch.exp(torch.tensor(avg_loss)).item(), 1e6)
    model.train()
    return avg_loss, perplexity


def train(model_config: ModelConfig, train_config: TrainConfig):
    device = get_device(train_config.device)
    print(f"Device: {device}")

    model_config.vocab_size = VOCAB_SIZE
    model = WorldLLM(model_config).to(device)
    print(f"Model parameters: {model.count_parameters():,}")

    train_loader = create_dataloader(
        train_config.train_path,
        max_seq_len=model_config.max_seq_len,
        batch_size=train_config.batch_size,
        shuffle=True,
    )
    val_loader = create_dataloader(
        train_config.val_path,
        max_seq_len=model_config.max_seq_len,
        batch_size=train_config.batch_size,
        shuffle=False,
    )

    print(f"Train examples: {len(train_loader.dataset):,}")
    print(f"Val examples: {len(val_loader.dataset):,}")
    print()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )

    os.makedirs(train_config.save_dir, exist_ok=True)
    best_val_loss = float("inf")

    epoch = 0
    while True:
        epoch += 1
        model.train()
        epoch_loss = 0.0
        epoch_tokens = 0
        t0 = time.time()

        for inputs, targets, loss_mask in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            loss_mask = loss_mask.to(device)

            logits = model(inputs)
            loss, n = masked_loss(logits, targets, loss_mask)
            if n > 0:
                loss = loss / n
            else:
                continue

            optimizer.zero_grad()
            loss.backward()
            if train_config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
            optimizer.step()

            epoch_loss += loss.item() * n
            epoch_tokens += n

        elapsed = time.time() - t0
        train_loss = epoch_loss / max(epoch_tokens, 1)
        val_loss, val_ppl = evaluate(model, val_loader, device)

        print(f"Epoch {epoch} | "
              f"train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | "
              f"val_ppl {val_ppl:.2f} | {elapsed:.1f}s")

        # Save latest checkpoint
        path = os.path.join(train_config.save_dir, "latest.pt")
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": model_config,
            "epoch": epoch,
            "val_loss": val_loss,
        }, path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(train_config.save_dir, "best.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": model_config,
                "epoch": epoch,
                "val_loss": val_loss,
            }, best_path)
            print(f"  -> new best model (val_loss {val_loss:.4f})")

    print(f"\nTraining complete. Best val_loss: {best_val_loss:.4f}")
    return model


def main():
    parser = argparse.ArgumentParser(description="Train WorldLLM")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--embed_dim", type=int, default=None)
    parser.add_argument("--num_layers", type=int, default=None)
    parser.add_argument("--num_heads", type=int, default=None)
    parser.add_argument("--train_path", type=str, default=None)
    parser.add_argument("--val_path", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    model_config = ModelConfig()
    train_config = TrainConfig()

    if args.epochs is not None:
        train_config.epochs = args.epochs
    if args.batch_size is not None:
        train_config.batch_size = args.batch_size
    if args.lr is not None:
        train_config.learning_rate = args.lr
    if args.embed_dim is not None:
        model_config.embed_dim = args.embed_dim
    if args.num_layers is not None:
        model_config.num_layers = args.num_layers
    if args.num_heads is not None:
        model_config.num_heads = args.num_heads
    if args.train_path is not None:
        train_config.train_path = args.train_path
    if args.val_path is not None:
        train_config.val_path = args.val_path
    if args.device is not None:
        train_config.device = args.device

    train(model_config, train_config)


if __name__ == "__main__":
    main()
