from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even.")

    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega

    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even.")

    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int, cls_token: bool = False) -> np.ndarray:
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def trunc_normal_(tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0, a: float = -2.0, b: float = 2.0):
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    with torch.no_grad():
        lower = norm_cdf((a - mean) / std)
        upper = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * lower - 1, 2 * upper - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


class DistanceEncoder(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(1, embed_dim // 4),
            nn.ReLU(),
            nn.Linear(embed_dim // 4, embed_dim),
        )

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        return self.encoder(distances)


class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim: int = 1024, num_heads: int = 8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, patch_features: torch.Tensor, retrieved_features: torch.Tensor) -> torch.Tensor:
        attn_output, _ = self.cross_attn(query=patch_features, key=retrieved_features, value=retrieved_features)
        return self.norm(patch_features + attn_output)


class SelfAttentionBlock(nn.Module):
    def __init__(self, embed_dim: int = 1024, num_heads: int = 8, num_patches: int = 256):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

        grid_size = int(num_patches**0.5)
        if grid_size * grid_size != num_patches:
            raise ValueError(f"num_patches must be a square number, got {num_patches}.")
        pos_embed = get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=True)
        self.register_buffer("pos_embedding", torch.from_numpy(pos_embed).float().unsqueeze(0), persistent=False)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features_with_pos = features + self.pos_embedding
        attn_output, _ = self.self_attn(query=features_with_pos, key=features_with_pos, value=features_with_pos)
        return self.norm(features + attn_output)


class ProcessingStream(nn.Module):
    def __init__(self, embed_dim: int = 1024, num_heads: int = 8, num_patches: int = 256, depth: int = 4):
        super().__init__()
        self.cross_attention_layers = nn.ModuleList([CrossAttentionBlock(embed_dim, num_heads) for _ in range(depth)])
        self.self_attention_layers = nn.ModuleList([SelfAttentionBlock(embed_dim, num_heads, num_patches) for _ in range(depth)])
        self.distance_encoder = DistanceEncoder(embed_dim)
        self.fusion_projection = nn.Linear(embed_dim * 2, embed_dim)
        self.num_patches = num_patches
        self.depth = depth

    def forward(self, features: torch.Tensor, retrieved_features: torch.Tensor, distances: torch.Tensor) -> torch.Tensor:
        batch_size, _, retrieved_count, embed_dim = retrieved_features.shape

        distance_embeddings = self.distance_encoder(distances.unsqueeze(-1))
        fused_kv_features = self.fusion_projection(torch.cat([retrieved_features, distance_embeddings], dim=-1))
        kv = fused_kv_features.reshape(batch_size * self.num_patches, retrieved_count, embed_dim)

        x = features
        for layer_idx in range(self.depth):
            cls_token = x[:, :1, :]
            patch_tokens = x[:, 1:, :]
            q = patch_tokens.reshape(batch_size * self.num_patches, 1, embed_dim)
            cross_attn_output = self.cross_attention_layers[layer_idx](q, kv)
            cross_attn_patches = cross_attn_output.reshape(batch_size, self.num_patches, embed_dim)
            x = self.self_attention_layers[layer_idx](torch.cat([cls_token, cross_attn_patches], dim=1))

        return x


class MainClassifier(nn.Module):
    def __init__(
        self,
        embed_dim: int = 1024,
        num_patches: int = 256,
        num_heads: int = 8,
        depth: int = 4,
        num_classes: int = 1,
        args=None,
    ):
        super().__init__()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        trunc_normal_(self.cls_token, std=0.02)

        self.normal_stream = ProcessingStream(embed_dim, num_heads, num_patches, depth=depth)
        self.abnormal_stream = ProcessingStream(embed_dim, num_heads, num_patches, depth=depth)
        self.fusion_head = nn.Sequential(
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(embed_dim, num_classes),
        )

        self.apply(self._initialize_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embedding", "cls_token"}

    def _initialize_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward(
        self,
        cur_features: torch.Tensor,
        cur_retrieved_features_from_normal: torch.Tensor,
        cur_distances_from_normal: torch.Tensor,
        cur_retrieved_features_from_abnormal: torch.Tensor,
        cur_distances_from_abnormal: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = cur_features.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        features_with_cls = torch.cat((cls_tokens, cur_features), dim=1)

        normal_output = self.normal_stream(
            features_with_cls,
            cur_retrieved_features_from_normal,
            cur_distances_from_normal,
        )
        abnormal_output = self.abnormal_stream(
            features_with_cls,
            cur_retrieved_features_from_abnormal,
            cur_distances_from_abnormal,
        )
        fused_cls = torch.cat([normal_output[:, 0], abnormal_output[:, 0]], dim=-1)
        return self.fusion_head(fused_cls)
