"""
Hyperparameters for WorldLLM.
"""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 137
    max_seq_len: int = 256
    embed_dim: int = 256
    num_heads: int = 8
    num_layers: int = 8
    ffn_dim: int = 1024
    dropout: float = 0.1


@dataclass
class TrainConfig:
    train_path: str = "data/train.txt"
    val_path: str = "data/val.txt"
    batch_size: int = 64
    learning_rate: float = 3e-3
    weight_decay: float = 0.01
    epochs: int = 20
    grad_clip: float = 1.0
    save_dir: str = "checkpoints"
    device: str = "auto"
