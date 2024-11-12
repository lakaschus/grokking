# model.py

from einops import rearrange, repeat
import torch
from torch import nn, Tensor


class DecoderBlock(nn.Module):
    def __init__(self, dim_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim_model, n_heads)
        self.self_attn_norm = nn.LayerNorm(dim_model)
        self.self_attn_drop = nn.Dropout(p=dropout)
        self.ffn = self._build_ffn(dim_model)
        self.ffn_drop = nn.Dropout(p=dropout)
        self.ffn_norm = nn.LayerNorm(dim_model)

    def _build_ffn(self, dim_model: int) -> nn.Sequential:

        return nn.Sequential(
            nn.Linear(dim_model, dim_model * 4),
            nn.ReLU(),
            nn.Linear(dim_model * 4, dim_model),
        )

    def forward(self, x: Tensor) -> Tensor:
        attn_mask = self._create_attn_mask(len(x), x.device, x.dtype)
        a1, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
        a1 = self.self_attn_drop(a1)
        a1 = self.self_attn_norm(x + a1)
        a2 = self.ffn(a1)
        a2 = self.ffn_drop(a2)
        a2 = self.ffn_norm(a1 + a2)
        return a2

    def _create_attn_mask(
        self, size: int, device: torch.device, dtype: torch.dtype
    ) -> Tensor:
        attn_mask = torch.full((size, size), -float("Inf"), device=device, dtype=dtype)
        return torch.triu(attn_mask, diagonal=1)


class Transformer(nn.Module):
    def __init__(
        self,
        config,
        num_layers: int,
        dim_model: int,
        num_heads: int,
        num_tokens: int,
        seq_len: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.config = config
        self.id_to_token = {k: str(k) for k in range(num_tokens)}
        self.id_to_token[config.eq_token] = " = "
        self.id_to_token[config.op_token] = " OP "
        self.dropout = dropout
        self.token_embeddings = nn.Embedding(num_tokens, dim_model)
        self.position_embeddings = nn.Embedding(seq_len, dim_model)
        self.decoder = self._build_decoder(num_layers, dim_model, num_heads)
        self.output_layer = nn.Sequential(
            nn.LayerNorm(dim_model), nn.Linear(dim_model, num_tokens)
        )

    def _build_decoder(
        self, num_layers: int, dim_model: int, num_heads: int
    ) -> nn.Sequential:
        return nn.Sequential(
            *[
                DecoderBlock(dim_model, num_heads, self.dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, inputs: Tensor) -> Tensor:

        embedding = self._create_embeddings(inputs)
        embedding = rearrange(embedding, "b s d -> s b d")
        encoded = self.decoder(embedding)
        return self.output_layer(encoded)

    def _create_embeddings(self, inputs: Tensor) -> Tensor:
        batch_size, context_len = inputs.shape
        token_embedding = self.token_embeddings(inputs)
        positions = repeat(
            torch.arange(context_len, device=inputs.device), "p -> b p", b=batch_size
        )
        position_embedding = self.position_embeddings(positions)
        return token_embedding + position_embedding
