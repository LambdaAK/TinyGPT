"""
Decoder-only transformer for TinyGPT.

Architecture (GPT-style, pre-norm):
    token embeddings + learned positional embeddings
    -> N x [LayerNorm -> CausalSelfAttention -> LayerNorm -> FFN] (residual connections)
    -> LayerNorm -> linear head (weight-tied with token embeddings)

The model is ~6.4M parameters with default config and trains in ~3 minutes on A100.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import ModelConfig


class CausalSelfAttention(nn.Module):
    """Multi-head causal (masked) self-attention.

    Uses a fused QKV projection for efficiency and delegates to
    F.scaled_dot_product_attention, which automatically dispatches to
    FlashAttention-2 on supported hardware (A100, H100).
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.embed_dim % config.num_heads == 0

        self.num_heads = config.num_heads
        self.head_dim = config.embed_dim // config.num_heads
        self.dropout = config.dropout

        self.qkv = nn.Linear(config.embed_dim, 3 * config.embed_dim)
        self.out_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        """
        Args:
            x: (B, T, C) input tensor
        Returns:
            (B, T, C) attention output
        """
        B, T, C = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.out_proj(out))


class FeedForward(nn.Module):
    """Position-wise two-layer FFN with GELU activation (embed_dim -> ffn_dim -> embed_dim)."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.embed_dim, config.ffn_dim),
            nn.GELU(),
            nn.Linear(config.ffn_dim, config.embed_dim),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: LN -> Attention -> residual -> LN -> FFN -> residual."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.embed_dim)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.embed_dim)
        self.ffn = FeedForward(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    """Decoder-only transformer for possession-tracking conversations.

    Consumes tokenized CLIENT:/OUTPUT: conversations and predicts the next token
    autoregressively. The output head shares weights with the token embedding
    table (weight tying) to reduce parameter count and improve generalization.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.embed_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.embed_dim)
        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])

        self.ln_f = nn.LayerNorm(config.embed_dim)
        self.head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

        # Weight tying: share token embedding weights with output head
        self.head.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self):
        """Initialize all linear and embedding layers with small normal weights (std=0.02)."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()

    def forward(self, input_ids):
        """
        input_ids: (B, T) token indices
        Returns logits: (B, T, vocab_size)
        """
        B, T = input_ids.shape
        assert T <= self.config.max_seq_len, f"Sequence length {T} exceeds max {self.config.max_seq_len}"

        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.drop(self.token_emb(input_ids) + self.pos_emb(positions))

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)
        return logits

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=50, temperature=1.0, top_k=None):
        """Autoregressive next-token generation with temperature and optional top-k sampling.

        Appends tokens one at a time until <eos> is produced or max_new_tokens
        is reached. When the sequence exceeds max_seq_len, only the most recent
        tokens are fed to the model (sliding window).

        Args:
            input_ids: (1, T) prompt token indices
            max_new_tokens: generation budget
            temperature: softmax temperature (lower = more deterministic)
            top_k: if set, restrict sampling to the top-k most probable tokens

        Returns:
            (1, T + generated) full sequence including prompt and generated tokens
        """
        self.eval()
        for _ in range(max_new_tokens):
            ids = input_ids if input_ids.size(1) <= self.config.max_seq_len else input_ids[:, -self.config.max_seq_len:]

            logits = self.forward(ids)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Stop on <eos>
            if next_token.item() == 2:  # EOS_ID
                break

        return input_ids

    def count_parameters(self):
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
