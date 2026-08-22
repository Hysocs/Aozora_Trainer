# Copyright 2026 Aozora Trainer contributors
# Portions derived from ModelScope DiffSynth-Studio 2.0.16 and modified for Aozora.
# Licensed under the Apache License, Version 2.0. See the repository LICENSE.
# SPDX-License-Identifier: Apache-2.0

# original code from: comfy/ldm/cosmos/predict2.py

import logging
import math
from typing import Callable, List, Optional, Tuple

import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn
from torchvision import transforms

try:
    from flash_attn import flash_attn_func
except ImportError:
    flash_attn_func = None


ANIMA_TEXT_SEQUENCE_LENGTH = 512


def gradient_checkpoint_forward(
    model, use_gradient_checkpointing, use_gradient_checkpointing_offload, *args, **kwargs
):
    if use_gradient_checkpointing_offload:
        with torch.autograd.graph.save_on_cpu():
            return torch.utils.checkpoint.checkpoint(
                model, *args, **kwargs, use_reentrant=False
            )
    if use_gradient_checkpointing:
        return torch.utils.checkpoint.checkpoint(
            model, *args, **kwargs, use_reentrant=False
        )
    return model(*args, **kwargs)


class VideoPositionEmb(nn.Module):
    def forward(self, x_B_T_H_W_C: torch.Tensor, fps=Optional[torch.Tensor], device=None, dtype=None) -> torch.Tensor:
        B_T_H_W_C = x_B_T_H_W_C.shape
        return self.generate_embeddings(B_T_H_W_C, fps=fps, device=device, dtype=dtype)

    def generate_embeddings(self, B_T_H_W_C: torch.Size, fps=Optional[torch.Tensor], device=None):
        raise NotImplementedError


def normalize(x: torch.Tensor, dim: Optional[List[int]] = None, eps: float = 0) -> torch.Tensor:
    if dim is None:
        dim = list(range(1, x.ndim))
    norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True, dtype=torch.float32)
    norm = torch.add(eps, norm, alpha=math.sqrt(norm.numel() / x.numel()))
    return x / norm.to(x.dtype)


class LearnablePosEmbAxis(VideoPositionEmb):
    def __init__(
        self,
        *,
        interpolation: str,
        model_channels: int,
        len_h: int,
        len_w: int,
        len_t: int,
        device=None,
        dtype=None,
        **kwargs,
    ):
        del kwargs
        super().__init__()
        self.interpolation = interpolation
        assert self.interpolation in ["crop"], f"Unknown interpolation method {self.interpolation}"

        self.pos_emb_h = nn.Parameter(torch.empty(len_h, model_channels, device=device, dtype=dtype))
        self.pos_emb_w = nn.Parameter(torch.empty(len_w, model_channels, device=device, dtype=dtype))
        self.pos_emb_t = nn.Parameter(torch.empty(len_t, model_channels, device=device, dtype=dtype))

    def generate_embeddings(self, B_T_H_W_C: torch.Size, fps=Optional[torch.Tensor], device=None, dtype=None) -> torch.Tensor:
        B, T, H, W, _ = B_T_H_W_C
        if self.interpolation != "crop":
            raise ValueError(f"Unknown interpolation method {self.interpolation}")

        emb_h_H = self.pos_emb_h[:H].to(device=device, dtype=dtype)
        emb_w_W = self.pos_emb_w[:W].to(device=device, dtype=dtype)
        emb_t_T = self.pos_emb_t[:T].to(device=device, dtype=dtype)
        emb = (
            repeat(emb_t_T, "t d-> b t h w d", b=B, h=H, w=W)
            + repeat(emb_h_H, "h d-> b t h w d", b=B, t=T, w=W)
            + repeat(emb_w_W, "w d-> b t h w d", b=B, t=T, h=H)
        )
        assert list(emb.shape)[:4] == [B, T, H, W], f"bad shape: {list(emb.shape)[:4]} != {B, T, H, W}"
        return normalize(emb, dim=-1, eps=1e-6)


class VideoRopePosition3DEmb(VideoPositionEmb):
    def __init__(
        self,
        *,
        head_dim: int,
        len_h: int,
        len_w: int,
        len_t: int,
        base_fps: int = 24,
        h_extrapolation_ratio: float = 1.0,
        w_extrapolation_ratio: float = 1.0,
        t_extrapolation_ratio: float = 1.0,
        enable_fps_modulation: bool = True,
        device=None,
        **kwargs,
    ):
        del kwargs
        super().__init__()
        self.base_fps = base_fps
        self.max_h = len_h
        self.max_w = len_w
        self.enable_fps_modulation = enable_fps_modulation

        dim = head_dim
        dim_h = dim // 6 * 2
        dim_w = dim_h
        dim_t = dim - 2 * dim_h
        assert dim == dim_h + dim_w + dim_t, f"bad dim: {dim} != {dim_h} + {dim_w} + {dim_t}"
        self.register_buffer(
            "dim_spatial_range",
            torch.arange(0, dim_h, 2, device=device)[: (dim_h // 2)].float() / dim_h,
            persistent=False,
        )
        self.register_buffer(
            "dim_temporal_range",
            torch.arange(0, dim_t, 2, device=device)[: (dim_t // 2)].float() / dim_t,
            persistent=False,
        )

        self.h_ntk_factor = h_extrapolation_ratio ** (dim_h / (dim_h - 2))
        self.w_ntk_factor = w_extrapolation_ratio ** (dim_w / (dim_w - 2))
        self.t_ntk_factor = t_extrapolation_ratio ** (dim_t / (dim_t - 2))

    def generate_embeddings(
        self,
        B_T_H_W_C: torch.Size,
        fps: Optional[torch.Tensor] = None,
        h_ntk_factor: Optional[float] = None,
        w_ntk_factor: Optional[float] = None,
        t_ntk_factor: Optional[float] = None,
        device=None,
        dtype=None,
    ):
        h_ntk_factor = h_ntk_factor if h_ntk_factor is not None else self.h_ntk_factor
        w_ntk_factor = w_ntk_factor if w_ntk_factor is not None else self.w_ntk_factor
        t_ntk_factor = t_ntk_factor if t_ntk_factor is not None else self.t_ntk_factor

        h_theta = 10000.0 * h_ntk_factor
        w_theta = 10000.0 * w_ntk_factor
        t_theta = 10000.0 * t_ntk_factor

        h_spatial_freqs = 1.0 / (h_theta ** self.dim_spatial_range.to(device=device))
        w_spatial_freqs = 1.0 / (w_theta ** self.dim_spatial_range.to(device=device))
        temporal_freqs = 1.0 / (t_theta ** self.dim_temporal_range.to(device=device))

        B, T, H, W, _ = B_T_H_W_C
        seq = torch.arange(max(H, W, T), dtype=torch.float, device=device)
        uniform_fps = (fps is None) or isinstance(fps, (int, float)) or (fps.min() == fps.max())
        assert uniform_fps or B == 1 or T == 1, (
            "For video batch, batch size should be 1 for non-uniform fps. "
            "For image batch, T should be 1"
        )
        half_emb_h = torch.outer(seq[:H], h_spatial_freqs)
        half_emb_w = torch.outer(seq[:W], w_spatial_freqs)

        if fps is None or self.enable_fps_modulation is False:
            half_emb_t = torch.outer(seq[:T], temporal_freqs)
        else:
            half_emb_t = torch.outer(seq[:T] / fps * self.base_fps, temporal_freqs)

        half_emb_h = torch.stack(
            [torch.cos(half_emb_h), -torch.sin(half_emb_h), torch.sin(half_emb_h), torch.cos(half_emb_h)], dim=-1
        )
        half_emb_w = torch.stack(
            [torch.cos(half_emb_w), -torch.sin(half_emb_w), torch.sin(half_emb_w), torch.cos(half_emb_w)], dim=-1
        )
        half_emb_t = torch.stack(
            [torch.cos(half_emb_t), -torch.sin(half_emb_t), torch.sin(half_emb_t), torch.cos(half_emb_t)], dim=-1
        )

        em_T_H_W_D = torch.cat(
            [
                repeat(half_emb_t, "t d x -> t h w d x", h=H, w=W),
                repeat(half_emb_h, "h d x -> t h w d x", t=T, w=W),
                repeat(half_emb_w, "w d x -> t h w d x", t=T, h=H),
            ],
            dim=-2,
        )
        return rearrange(em_T_H_W_D, "t h w d (i j) -> (t h w) d i j", i=2, j=2).float()


def apply_rotary_pos_emb(t: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    t_ = t.reshape(*t.shape[:-1], 2, -1).movedim(-2, -1).unsqueeze(-2).float()
    t_out = freqs[..., 0] * t_[..., 0] + freqs[..., 1] * t_[..., 1]
    return t_out.movedim(-1, -2).reshape(*t.shape).type_as(t)


class GPT2FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None, operations=None) -> None:
        super().__init__()
        self.activation = nn.GELU()
        self.layer1 = operations.Linear(d_model, d_ff, bias=False, device=device, dtype=dtype)
        self.layer2 = operations.Linear(d_ff, d_model, bias=False, device=device, dtype=dtype)
        self._layer_id = None
        self._dim = d_model
        self._hidden_dim = d_ff

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer2(self.activation(self.layer1(x)))


def torch_attention_op(
    q_B_S_H_D: torch.Tensor,
    k_B_S_H_D: torch.Tensor,
    v_B_S_H_D: torch.Tensor,
    transformer_options: Optional[dict] = {},
) -> torch.Tensor:
    in_q_shape = q_B_S_H_D.shape
    in_k_shape = k_B_S_H_D.shape
    q = q_B_S_H_D.reshape(in_q_shape[0], -1, in_q_shape[-2], in_q_shape[-1])
    k = k_B_S_H_D.reshape(in_k_shape[0], -1, in_k_shape[-2], in_k_shape[-1])
    v = v_B_S_H_D.reshape(in_k_shape[0], -1, in_k_shape[-2], in_k_shape[-1])
    if flash_attn_func is not None and q.is_cuda and q.dtype in (torch.float16, torch.bfloat16):
        output = flash_attn_func(q, k, v)
    else:
        output = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        ).transpose(1, 2)
    return rearrange(output, "b s n d -> b s (n d)")


class Attention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        context_dim: Optional[int] = None,
        n_heads: int = 8,
        head_dim: int = 64,
        dropout: float = 0.0,
        device=None,
        dtype=None,
        operations=None,
    ) -> None:
        super().__init__()
        logging.debug(
            f"Setting up {self.__class__.__name__}. Query dim is {query_dim}, context_dim is {context_dim} and using "
            f"{n_heads} heads with a dimension of {head_dim}."
        )
        self.is_selfattn = context_dim is None
        context_dim = query_dim if context_dim is None else context_dim
        inner_dim = head_dim * n_heads

        self.n_heads = n_heads
        self.head_dim = head_dim
        self.query_dim = query_dim
        self.context_dim = context_dim

        self.q_proj = operations.Linear(query_dim, inner_dim, bias=False, device=device, dtype=dtype)
        self.q_norm = operations.RMSNorm(self.head_dim, eps=1e-6, device=device, dtype=dtype)
        self.k_proj = operations.Linear(context_dim, inner_dim, bias=False, device=device, dtype=dtype)
        self.k_norm = operations.RMSNorm(self.head_dim, eps=1e-6, device=device, dtype=dtype)
        self.v_proj = operations.Linear(context_dim, inner_dim, bias=False, device=device, dtype=dtype)
        self.v_norm = nn.Identity()
        self.output_proj = operations.Linear(inner_dim, query_dim, bias=False, device=device, dtype=dtype)
        self.output_dropout = nn.Dropout(dropout) if dropout > 1e-4 else nn.Identity()
        self.attn_op = torch_attention_op
        self._query_dim = query_dim
        self._context_dim = context_dim
        self._inner_dim = inner_dim

    def compute_qkv(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        rope_emb: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self.q_proj(x)
        context = x if context is None else context
        k = self.k_proj(context)
        v = self.v_proj(context)
        q, k, v = map(
            lambda t: rearrange(t, "b ... (h d) -> b ... h d", h=self.n_heads, d=self.head_dim),
            (q, k, v),
        )
        q = self.q_norm(q)
        k = self.k_norm(k)
        v = self.v_norm(v)
        if self.is_selfattn and rope_emb is not None:
            q = apply_rotary_pos_emb(q, rope_emb)
            k = apply_rotary_pos_emb(k, rope_emb)
        return q, k, v

    def compute_attention(self, q, k, v, transformer_options: Optional[dict] = {}) -> torch.Tensor:
        result = self.attn_op(q, k, v, transformer_options=transformer_options)
        return self.output_dropout(self.output_proj(result))

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        rope_emb: Optional[torch.Tensor] = None,
        transformer_options: Optional[dict] = {},
    ) -> torch.Tensor:
        q, k, v = self.compute_qkv(x, context, rope_emb=rope_emb)
        return self.compute_attention(q, k, v, transformer_options=transformer_options)


class Timesteps(nn.Module):
    def __init__(self, num_channels: int):
        super().__init__()
        self.num_channels = num_channels

    def forward(self, timesteps_B_T: torch.Tensor) -> torch.Tensor:
        assert timesteps_B_T.ndim == 2, f"Expected 2D input, got {timesteps_B_T.ndim}"
        timesteps = timesteps_B_T.flatten().float()
        half_dim = self.num_channels // 2
        exponent = -math.log(10000) * torch.arange(half_dim, dtype=torch.float32, device=timesteps.device)
        exponent = exponent / half_dim
        emb = timesteps[:, None].float() * torch.exp(exponent)[None, :]
        emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)
        return rearrange(emb, "(b t) d -> b t d", b=timesteps_B_T.shape[0], t=timesteps_B_T.shape[1])


class TimestepEmbedding(nn.Module):
    def __init__(self, in_features: int, out_features: int, use_adaln_lora: bool = False, device=None, dtype=None, operations=None):
        super().__init__()
        self.in_dim = in_features
        self.out_dim = out_features
        self.linear_1 = operations.Linear(in_features, out_features, bias=not use_adaln_lora, device=device, dtype=dtype)
        self.activation = nn.SiLU()
        self.use_adaln_lora = use_adaln_lora
        if use_adaln_lora:
            self.linear_2 = operations.Linear(out_features, 3 * out_features, bias=False, device=device, dtype=dtype)
        else:
            self.linear_2 = operations.Linear(out_features, out_features, bias=False, device=device, dtype=dtype)

    def forward(self, sample: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        emb = self.linear_2(self.activation(self.linear_1(sample)))
        if self.use_adaln_lora:
            return sample, emb
        return emb, None


class PatchEmbed(nn.Module):
    def __init__(
        self,
        spatial_patch_size: int,
        temporal_patch_size: int,
        in_channels: int = 3,
        out_channels: int = 768,
        device=None,
        dtype=None,
        operations=None,
    ):
        super().__init__()
        self.spatial_patch_size = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size
        self.proj = nn.Sequential(
            Rearrange(
                "b c (t r) (h m) (w n) -> b t h w (c r m n)",
                r=temporal_patch_size,
                m=spatial_patch_size,
                n=spatial_patch_size,
            ),
            operations.Linear(
                in_channels * spatial_patch_size * spatial_patch_size * temporal_patch_size,
                out_channels,
                bias=False,
                device=device,
                dtype=dtype,
            ),
        )
        self.dim = in_channels * spatial_patch_size * spatial_patch_size * temporal_patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 5
        _, _, T, H, W = x.shape
        assert H % self.spatial_patch_size == 0 and W % self.spatial_patch_size == 0
        assert T % self.temporal_patch_size == 0
        return self.proj(x)


class FinalLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        spatial_patch_size: int,
        temporal_patch_size: int,
        out_channels: int,
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
        device=None,
        dtype=None,
        operations=None,
    ):
        super().__init__()
        self.layer_norm = operations.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = operations.Linear(
            hidden_size,
            spatial_patch_size * spatial_patch_size * temporal_patch_size * out_channels,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.hidden_size = hidden_size
        self.n_adaln_chunks = 2
        self.use_adaln_lora = use_adaln_lora
        self.adaln_lora_dim = adaln_lora_dim
        if use_adaln_lora:
            self.adaln_modulation = nn.Sequential(
                nn.SiLU(),
                operations.Linear(hidden_size, adaln_lora_dim, bias=False, device=device, dtype=dtype),
                operations.Linear(adaln_lora_dim, 2 * hidden_size, bias=False, device=device, dtype=dtype),
            )
        else:
            self.adaln_modulation = nn.Sequential(
                nn.SiLU(), operations.Linear(hidden_size, 2 * hidden_size, bias=False, device=device, dtype=dtype)
            )

    def forward(self, x_B_T_H_W_D, emb_B_T_D, adaln_lora_B_T_3D=None):
        if self.use_adaln_lora:
            assert adaln_lora_B_T_3D is not None
            shift_B_T_D, scale_B_T_D = (
                self.adaln_modulation(emb_B_T_D) + adaln_lora_B_T_3D[:, :, : 2 * self.hidden_size]
            ).chunk(2, dim=-1)
        else:
            shift_B_T_D, scale_B_T_D = self.adaln_modulation(emb_B_T_D).chunk(2, dim=-1)
        shift = rearrange(shift_B_T_D, "b t d -> b t 1 1 d")
        scale = rearrange(scale_B_T_D, "b t d -> b t 1 1 d")
        x_B_T_H_W_D = self.layer_norm(x_B_T_H_W_D) * (1 + scale) + shift
        return self.linear(x_B_T_H_W_D)


class Block(nn.Module):
    def __init__(
        self,
        x_dim: int,
        context_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
        device=None,
        dtype=None,
        operations=None,
    ):
        super().__init__()
        self.x_dim = x_dim
        self.layer_norm_self_attn = operations.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6, device=device, dtype=dtype)
        self.self_attn = Attention(x_dim, None, num_heads, x_dim // num_heads, device=device, dtype=dtype, operations=operations)
        self.layer_norm_cross_attn = operations.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6, device=device, dtype=dtype)
        self.cross_attn = Attention(x_dim, context_dim, num_heads, x_dim // num_heads, device=device, dtype=dtype, operations=operations)
        self.layer_norm_mlp = operations.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6, device=device, dtype=dtype)
        self.mlp = GPT2FeedForward(x_dim, int(x_dim * mlp_ratio), device=device, dtype=dtype, operations=operations)
        self.use_adaln_lora = use_adaln_lora
        if use_adaln_lora:
            def modulation():
                return nn.Sequential(
                    nn.SiLU(),
                    operations.Linear(x_dim, adaln_lora_dim, bias=False, device=device, dtype=dtype),
                    operations.Linear(adaln_lora_dim, 3 * x_dim, bias=False, device=device, dtype=dtype),
                )
            self.adaln_modulation_self_attn = modulation()
            self.adaln_modulation_cross_attn = modulation()
            self.adaln_modulation_mlp = modulation()
        else:
            self.adaln_modulation_self_attn = nn.Sequential(nn.SiLU(), operations.Linear(x_dim, 3 * x_dim, bias=False, device=device, dtype=dtype))
            self.adaln_modulation_cross_attn = nn.Sequential(nn.SiLU(), operations.Linear(x_dim, 3 * x_dim, bias=False, device=device, dtype=dtype))
            self.adaln_modulation_mlp = nn.Sequential(nn.SiLU(), operations.Linear(x_dim, 3 * x_dim, bias=False, device=device, dtype=dtype))

    def forward(
        self,
        x_B_T_H_W_D: torch.Tensor,
        emb_B_T_D: torch.Tensor,
        crossattn_emb: torch.Tensor,
        rope_emb_L_1_1_D: Optional[torch.Tensor] = None,
        adaln_lora_B_T_3D: Optional[torch.Tensor] = None,
        extra_per_block_pos_emb: Optional[torch.Tensor] = None,
        transformer_options: Optional[dict] = {},
    ) -> torch.Tensor:
        residual_dtype = x_B_T_H_W_D.dtype
        compute_dtype = emb_B_T_D.dtype
        if extra_per_block_pos_emb is not None:
            x_B_T_H_W_D = x_B_T_H_W_D + extra_per_block_pos_emb

        if self.use_adaln_lora:
            self_shift, self_scale, self_gate = (self.adaln_modulation_self_attn(emb_B_T_D) + adaln_lora_B_T_3D).chunk(3, dim=-1)
            cross_shift, cross_scale, cross_gate = (self.adaln_modulation_cross_attn(emb_B_T_D) + adaln_lora_B_T_3D).chunk(3, dim=-1)
            mlp_shift, mlp_scale, mlp_gate = (self.adaln_modulation_mlp(emb_B_T_D) + adaln_lora_B_T_3D).chunk(3, dim=-1)
        else:
            self_shift, self_scale, self_gate = self.adaln_modulation_self_attn(emb_B_T_D).chunk(3, dim=-1)
            cross_shift, cross_scale, cross_gate = self.adaln_modulation_cross_attn(emb_B_T_D).chunk(3, dim=-1)
            mlp_shift, mlp_scale, mlp_gate = self.adaln_modulation_mlp(emb_B_T_D).chunk(3, dim=-1)

        def expand(t):
            return rearrange(t, "b t d -> b t 1 1 d")

        self_shift, self_scale, self_gate = map(expand, (self_shift, self_scale, self_gate))
        cross_shift, cross_scale, cross_gate = map(expand, (cross_shift, cross_scale, cross_gate))
        mlp_shift, mlp_scale, mlp_gate = map(expand, (mlp_shift, mlp_scale, mlp_gate))
        B, T, H, W, _ = x_B_T_H_W_D.shape

        normalized = self.layer_norm_self_attn(x_B_T_H_W_D) * (1 + self_scale) + self_shift
        result = rearrange(
            self.self_attn(
                rearrange(normalized.to(compute_dtype), "b t h w d -> b (t h w) d"),
                None,
                rope_emb=rope_emb_L_1_1_D,
                transformer_options=transformer_options,
            ),
            "b (t h w) d -> b t h w d",
            t=T,
            h=H,
            w=W,
        )
        x_B_T_H_W_D = x_B_T_H_W_D + self_gate.to(residual_dtype) * result.to(residual_dtype)

        normalized = self.layer_norm_cross_attn(x_B_T_H_W_D) * (1 + cross_scale) + cross_shift
        result = rearrange(
            self.cross_attn(
                rearrange(normalized.to(compute_dtype), "b t h w d -> b (t h w) d"),
                crossattn_emb,
                rope_emb=rope_emb_L_1_1_D,
                transformer_options=transformer_options,
            ),
            "b (t h w) d -> b t h w d",
            t=T,
            h=H,
            w=W,
        )
        x_B_T_H_W_D = x_B_T_H_W_D + cross_gate.to(residual_dtype) * result.to(residual_dtype)

        normalized = self.layer_norm_mlp(x_B_T_H_W_D) * (1 + mlp_scale) + mlp_shift
        result = self.mlp(normalized.to(compute_dtype))
        return x_B_T_H_W_D + mlp_gate.to(residual_dtype) * result.to(residual_dtype)


class MiniTrainDIT(nn.Module):
    def __init__(
        self,
        max_img_h: int,
        max_img_w: int,
        max_frames: int,
        in_channels: int,
        out_channels: int,
        patch_spatial: int,
        patch_temporal: int,
        concat_padding_mask: bool = True,
        model_channels: int = 768,
        num_blocks: int = 10,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        crossattn_emb_channels: int = 1024,
        pos_emb_cls: str = "sincos",
        pos_emb_learnable: bool = False,
        pos_emb_interpolation: str = "crop",
        min_fps: int = 1,
        max_fps: int = 30,
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
        rope_h_extrapolation_ratio: float = 1.0,
        rope_w_extrapolation_ratio: float = 1.0,
        rope_t_extrapolation_ratio: float = 1.0,
        extra_per_block_abs_pos_emb: bool = False,
        extra_h_extrapolation_ratio: float = 1.0,
        extra_w_extrapolation_ratio: float = 1.0,
        extra_t_extrapolation_ratio: float = 1.0,
        rope_enable_fps_modulation: bool = True,
        image_model=None,
        device=None,
        dtype=None,
        operations=None,
    ) -> None:
        super().__init__()
        self.dtype = dtype
        self.max_img_h = max_img_h
        self.max_img_w = max_img_w
        self.max_frames = max_frames
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_spatial = patch_spatial
        self.patch_temporal = patch_temporal
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.model_channels = model_channels
        self.concat_padding_mask = concat_padding_mask
        self.pos_emb_cls = pos_emb_cls
        self.pos_emb_learnable = pos_emb_learnable
        self.pos_emb_interpolation = pos_emb_interpolation
        self.min_fps = min_fps
        self.max_fps = max_fps
        self.rope_h_extrapolation_ratio = rope_h_extrapolation_ratio
        self.rope_w_extrapolation_ratio = rope_w_extrapolation_ratio
        self.rope_t_extrapolation_ratio = rope_t_extrapolation_ratio
        self.extra_per_block_abs_pos_emb = extra_per_block_abs_pos_emb
        self.extra_h_extrapolation_ratio = extra_h_extrapolation_ratio
        self.extra_w_extrapolation_ratio = extra_w_extrapolation_ratio
        self.extra_t_extrapolation_ratio = extra_t_extrapolation_ratio
        self.rope_enable_fps_modulation = rope_enable_fps_modulation

        self.build_pos_embed(device=device, dtype=dtype)
        self.use_adaln_lora = use_adaln_lora
        self.adaln_lora_dim = adaln_lora_dim
        self.t_embedder = nn.Sequential(
            Timesteps(model_channels),
            TimestepEmbedding(model_channels, model_channels, use_adaln_lora=use_adaln_lora, device=device, dtype=dtype, operations=operations),
        )

        in_channels = in_channels + 1 if concat_padding_mask else in_channels
        self.x_embedder = PatchEmbed(
            spatial_patch_size=patch_spatial,
            temporal_patch_size=patch_temporal,
            in_channels=in_channels,
            out_channels=model_channels,
            device=device,
            dtype=dtype,
            operations=operations,
        )
        self.blocks = nn.ModuleList([
            Block(
                x_dim=model_channels,
                context_dim=crossattn_emb_channels,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                use_adaln_lora=use_adaln_lora,
                adaln_lora_dim=adaln_lora_dim,
                device=device,
                dtype=dtype,
                operations=operations,
            )
            for _ in range(num_blocks)
        ])
        self.final_layer = FinalLayer(
            hidden_size=model_channels,
            spatial_patch_size=patch_spatial,
            temporal_patch_size=patch_temporal,
            out_channels=out_channels,
            use_adaln_lora=use_adaln_lora,
            adaln_lora_dim=adaln_lora_dim,
            device=device,
            dtype=dtype,
            operations=operations,
        )
        self.t_embedding_norm = operations.RMSNorm(model_channels, eps=1e-6, device=device, dtype=dtype)

    def build_pos_embed(self, device=None, dtype=None) -> None:
        if self.pos_emb_cls != "rope3d":
            raise ValueError(f"Unknown pos_emb_cls {self.pos_emb_cls}")
        kwargs = dict(
            model_channels=self.model_channels,
            len_h=self.max_img_h // self.patch_spatial,
            len_w=self.max_img_w // self.patch_spatial,
            len_t=self.max_frames // self.patch_temporal,
            max_fps=self.max_fps,
            min_fps=self.min_fps,
            is_learnable=self.pos_emb_learnable,
            interpolation=self.pos_emb_interpolation,
            head_dim=self.model_channels // self.num_heads,
            h_extrapolation_ratio=self.rope_h_extrapolation_ratio,
            w_extrapolation_ratio=self.rope_w_extrapolation_ratio,
            t_extrapolation_ratio=self.rope_t_extrapolation_ratio,
            enable_fps_modulation=self.rope_enable_fps_modulation,
            device=device,
        )
        self.pos_embedder = VideoRopePosition3DEmb(**kwargs)
        if self.extra_per_block_abs_pos_emb:
            kwargs.update(
                h_extrapolation_ratio=self.extra_h_extrapolation_ratio,
                w_extrapolation_ratio=self.extra_w_extrapolation_ratio,
                t_extrapolation_ratio=self.extra_t_extrapolation_ratio,
                device=device,
                dtype=dtype,
            )
            self.extra_pos_embedder = LearnablePosEmbAxis(**kwargs)

    def prepare_embedded_sequence(self, x_B_C_T_H_W, fps=None, padding_mask=None):
        if self.concat_padding_mask:
            if padding_mask is None:
                padding_mask = torch.zeros(
                    x_B_C_T_H_W.shape[0], 1, x_B_C_T_H_W.shape[3], x_B_C_T_H_W.shape[4],
                    dtype=x_B_C_T_H_W.dtype, device=x_B_C_T_H_W.device,
                )
            else:
                padding_mask = transforms.functional.resize(
                    padding_mask, list(x_B_C_T_H_W.shape[-2:]), interpolation=transforms.InterpolationMode.NEAREST
                )
            x_B_C_T_H_W = torch.cat(
                [x_B_C_T_H_W, padding_mask.unsqueeze(1).repeat(1, 1, x_B_C_T_H_W.shape[2], 1, 1)], dim=1
            )
        x_B_T_H_W_D = self.x_embedder(x_B_C_T_H_W)
        extra_pos_emb = (
            self.extra_pos_embedder(x_B_T_H_W_D, fps=fps, device=x_B_C_T_H_W.device, dtype=x_B_C_T_H_W.dtype)
            if self.extra_per_block_abs_pos_emb else None
        )
        if "rope" in self.pos_emb_cls.lower():
            return x_B_T_H_W_D, self.pos_embedder(x_B_T_H_W_D, fps=fps, device=x_B_C_T_H_W.device), extra_pos_emb
        x_B_T_H_W_D = x_B_T_H_W_D + self.pos_embedder(x_B_T_H_W_D, device=x_B_C_T_H_W.device)
        return x_B_T_H_W_D, None, extra_pos_emb

    def unpatchify(self, x_B_T_H_W_M):
        return rearrange(
            x_B_T_H_W_M,
            "B T H W (p1 p2 t C) -> B C (T t) (H p1) (W p2)",
            p1=self.patch_spatial,
            p2=self.patch_spatial,
            t=self.patch_temporal,
        )

    def pad_to_patch_size(self, img, patch_size=(2, 2), padding_mode="circular"):
        if padding_mode == "circular" and (torch.jit.is_tracing() or torch.jit.is_scripting()):
            padding_mode = "reflect"
        pad = ()
        for i in range(img.ndim - 2):
            pad = (0, (patch_size[i] - img.shape[i + 2] % patch_size[i]) % patch_size[i]) + pad
        return torch.nn.functional.pad(img, pad, mode=padding_mode)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        use_gradient_checkpointing=False,
        use_gradient_checkpointing_offload=False,
        **kwargs,
    ):
        orig_shape = list(x.shape)
        x = self.pad_to_patch_size(x, (self.patch_temporal, self.patch_spatial, self.patch_spatial))
        x_B_T_H_W_D, rope_emb, extra_pos_emb = self.prepare_embedded_sequence(x, fps=fps, padding_mask=padding_mask)
        if timesteps.ndim == 1:
            timesteps = timesteps.unsqueeze(1)
        t_embedding, adaln_lora = self.t_embedder[1](self.t_embedder[0](timesteps).to(x_B_T_H_W_D.dtype))
        t_embedding = self.t_embedding_norm(t_embedding)
        self.affline_scale_log_info = {"t_embedding_B_T_D": t_embedding.detach()}
        self.affline_emb = t_embedding
        self.crossattn_emb = context

        block_kwargs = {
            "rope_emb_L_1_1_D": rope_emb.unsqueeze(1).unsqueeze(0),
            "adaln_lora_B_T_3D": adaln_lora,
            "extra_per_block_pos_emb": extra_pos_emb,
            "transformer_options": kwargs.get("transformer_options", {}),
        }
        if x_B_T_H_W_D.dtype == torch.float16:
            x_B_T_H_W_D = x_B_T_H_W_D.float()
        for block in self.blocks:
            x_B_T_H_W_D = gradient_checkpoint_forward(
                block,
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
                x_B_T_H_W_D=x_B_T_H_W_D,
                emb_B_T_D=t_embedding,
                crossattn_emb=context,
                **block_kwargs,
            )
        x_B_T_H_W_O = self.final_layer(x_B_T_H_W_D.to(context.dtype), t_embedding, adaln_lora_B_T_3D=adaln_lora)
        return self.unpatchify(x_B_T_H_W_O)[:, :, :orig_shape[-3], :orig_shape[-2], :orig_shape[-1]]


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb2(x, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (x * cos) + (rotate_half(x) * sin)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim):
        super().__init__()
        self.rope_theta = 10000
        inv_freq = 1.0 / (
            self.rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class LLMAdapterAttention(nn.Module):
    def __init__(self, query_dim, context_dim, n_heads, head_dim, device=None, dtype=None, operations=None):
        super().__init__()
        inner_dim = head_dim * n_heads
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.query_dim = query_dim
        self.context_dim = context_dim
        self.q_proj = operations.Linear(query_dim, inner_dim, bias=False, device=device, dtype=dtype)
        self.q_norm = operations.RMSNorm(self.head_dim, eps=1e-6, device=device, dtype=dtype)
        self.k_proj = operations.Linear(context_dim, inner_dim, bias=False, device=device, dtype=dtype)
        self.k_norm = operations.RMSNorm(self.head_dim, eps=1e-6, device=device, dtype=dtype)
        self.v_proj = operations.Linear(context_dim, inner_dim, bias=False, device=device, dtype=dtype)
        self.o_proj = operations.Linear(inner_dim, query_dim, bias=False, device=device, dtype=dtype)

    def forward(self, x, mask=None, context=None, position_embeddings=None, position_embeddings_context=None):
        context = x if context is None else context
        input_shape = x.shape[:-1]
        q_shape = (*input_shape, self.n_heads, self.head_dim)
        context_shape = context.shape[:-1]
        kv_shape = (*context_shape, self.n_heads, self.head_dim)
        query_states = self.q_norm(self.q_proj(x).view(q_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(context).view(kv_shape)).transpose(1, 2)
        value_states = self.v_proj(context).view(kv_shape).transpose(1, 2)
        if position_embeddings is not None:
            assert position_embeddings_context is not None
            cos, sin = position_embeddings
            query_states = apply_rotary_pos_emb2(query_states, cos, sin)
            cos, sin = position_embeddings_context
            key_states = apply_rotary_pos_emb2(key_states, cos, sin)
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states, key_states, value_states, attn_mask=mask
        )
        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output)

    def init_weights(self):
        torch.nn.init.zeros_(self.o_proj.weight)


class LLMAdapterTransformerBlock(nn.Module):
    def __init__(self, source_dim, model_dim, num_heads=16, mlp_ratio=4.0, use_self_attn=False, layer_norm=False, device=None, dtype=None, operations=None):
        super().__init__()
        self.use_self_attn = use_self_attn
        norm = lambda: operations.LayerNorm(model_dim, device=device, dtype=dtype) if layer_norm else operations.RMSNorm(model_dim, eps=1e-6, device=device, dtype=dtype)
        if use_self_attn:
            self.norm_self_attn = norm()
            self.self_attn = LLMAdapterAttention(model_dim, model_dim, num_heads, model_dim // num_heads, device=device, dtype=dtype, operations=operations)
        self.norm_cross_attn = norm()
        self.cross_attn = LLMAdapterAttention(model_dim, source_dim, num_heads, model_dim // num_heads, device=device, dtype=dtype, operations=operations)
        self.norm_mlp = norm()
        self.mlp = nn.Sequential(
            operations.Linear(model_dim, int(model_dim * mlp_ratio), device=device, dtype=dtype),
            nn.GELU(),
            operations.Linear(int(model_dim * mlp_ratio), model_dim, device=device, dtype=dtype),
        )

    def forward(self, x, context, target_attention_mask=None, source_attention_mask=None, position_embeddings=None, position_embeddings_context=None):
        if self.use_self_attn:
            x = x + self.self_attn(
                self.norm_self_attn(x),
                mask=target_attention_mask,
                position_embeddings=position_embeddings,
                position_embeddings_context=position_embeddings,
            )
        x = x + self.cross_attn(
            self.norm_cross_attn(x),
            mask=source_attention_mask,
            context=context,
            position_embeddings=position_embeddings,
            position_embeddings_context=position_embeddings_context,
        )
        return x + self.mlp(self.norm_mlp(x))

    def init_weights(self):
        torch.nn.init.zeros_(self.mlp[2].weight)
        self.cross_attn.init_weights()


class LLMAdapter(nn.Module):
    def __init__(
        self,
        source_dim=1024,
        target_dim=1024,
        model_dim=1024,
        num_layers=6,
        num_heads=16,
        use_self_attn=True,
        layer_norm=False,
        device=None,
        dtype=None,
        operations=None,
    ):
        super().__init__()
        self.embed = operations.Embedding(32128, target_dim, device=device, dtype=dtype)
        self.in_proj = operations.Linear(target_dim, model_dim, device=device, dtype=dtype) if model_dim != target_dim else nn.Identity()
        self.rotary_emb = RotaryEmbedding(model_dim // num_heads)
        self.blocks = nn.ModuleList([
            LLMAdapterTransformerBlock(
                source_dim, model_dim, num_heads=num_heads, use_self_attn=use_self_attn,
                layer_norm=layer_norm, device=device, dtype=dtype, operations=operations,
            )
            for _ in range(num_layers)
        ])
        self.out_proj = operations.Linear(model_dim, target_dim, device=device, dtype=dtype)
        self.norm = operations.RMSNorm(target_dim, eps=1e-6, device=device, dtype=dtype)

    def forward(self, source_hidden_states, target_input_ids, target_attention_mask=None, source_attention_mask=None):
        if target_attention_mask is not None:
            target_attention_mask = target_attention_mask.to(device=target_input_ids.device, dtype=torch.bool)
            if target_attention_mask.ndim == 2:
                target_attention_mask = target_attention_mask.unsqueeze(1).unsqueeze(1)
        if source_attention_mask is not None:
            source_attention_mask = source_attention_mask.to(device=source_hidden_states.device, dtype=torch.bool)
            if source_attention_mask.ndim == 2:
                source_attention_mask = source_attention_mask.unsqueeze(1).unsqueeze(1)

        context = source_hidden_states
        x = self.in_proj(self.embed(target_input_ids).to(context.dtype))
        position_ids = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        position_ids_context = torch.arange(context.shape[1], device=x.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(x, position_ids)
        position_embeddings_context = self.rotary_emb(x, position_ids_context)
        for block in self.blocks:
            x = block(
                x,
                context,
                target_attention_mask=target_attention_mask,
                source_attention_mask=source_attention_mask,
                position_embeddings=position_embeddings,
                position_embeddings_context=position_embeddings_context,
            )
        return self.norm(self.out_proj(x))


class AnimaDiT(MiniTrainDIT):
    _repeated_blocks = ["Block"]

    def __init__(self):
        kwargs = {
            'image_model': 'anima', 'max_img_h': 240, 'max_img_w': 240, 'max_frames': 128,
            'in_channels': 16, 'out_channels': 16, 'patch_spatial': 2, 'patch_temporal': 1,
            'model_channels': 2048, 'concat_padding_mask': True, 'crossattn_emb_channels': 1024,
            'pos_emb_cls': 'rope3d', 'pos_emb_learnable': True, 'pos_emb_interpolation': 'crop',
            'min_fps': 1, 'max_fps': 30, 'use_adaln_lora': True, 'adaln_lora_dim': 256,
            'num_blocks': 28, 'num_heads': 16, 'extra_per_block_abs_pos_emb': False,
            'rope_h_extrapolation_ratio': 4.0, 'rope_w_extrapolation_ratio': 4.0,
            'rope_t_extrapolation_ratio': 1.0, 'extra_h_extrapolation_ratio': 1.0,
            'extra_w_extrapolation_ratio': 1.0, 'extra_t_extrapolation_ratio': 1.0,
            'rope_enable_fps_modulation': False, 'dtype': torch.bfloat16, 'device': None,
            'operations': torch.nn,
        }
        super().__init__(**kwargs)
        self.llm_adapter = LLMAdapter(
            device=kwargs.get("device"), dtype=kwargs.get("dtype"), operations=kwargs.get("operations")
        )

    def preprocess_text_embeds(
        self,
        text_embeds,
        text_ids,
        qwen_attention_mask=None,
        t5xxl_attention_mask=None,
        t5xxl_weights=None,
    ):
        if text_ids is None:
            return text_embeds

        if qwen_attention_mask is not None:
            qwen_attention_mask = qwen_attention_mask.to(device=text_embeds.device, dtype=torch.bool)
            text_embeds = text_embeds * qwen_attention_mask.unsqueeze(-1).to(text_embeds.dtype)

        out = self.llm_adapter(
            text_embeds,
            text_ids,
            target_attention_mask=t5xxl_attention_mask,
            source_attention_mask=qwen_attention_mask,
        )
        if t5xxl_weights is not None:
            out = out * t5xxl_weights

        if t5xxl_attention_mask is not None:
            target_mask = t5xxl_attention_mask.to(device=out.device, dtype=torch.bool)
            out = out * target_mask.unsqueeze(-1).to(out.dtype)

        if out.shape[1] < ANIMA_TEXT_SEQUENCE_LENGTH:
            out = torch.nn.functional.pad(out, (0, 0, 0, ANIMA_TEXT_SEQUENCE_LENGTH - out.shape[1]))
        elif out.shape[1] > ANIMA_TEXT_SEQUENCE_LENGTH:
            out = out[:, :ANIMA_TEXT_SEQUENCE_LENGTH]
        return out

    def forward(
        self,
        x,
        timesteps,
        context,
        use_gradient_checkpointing=False,
        use_gradient_checkpointing_offload=False,
        **kwargs,
    ):
        t5xxl_ids = kwargs.pop("t5xxl_ids", None)
        qwen_attention_mask = kwargs.pop("qwen_attention_mask", None)
        t5xxl_attention_mask = kwargs.pop("t5xxl_attention_mask", None)
        if t5xxl_ids is not None:
            context = self.preprocess_text_embeds(
                context,
                t5xxl_ids,
                qwen_attention_mask=qwen_attention_mask,
                t5xxl_attention_mask=t5xxl_attention_mask,
                t5xxl_weights=kwargs.pop("t5xxl_weights", None),
            )
        return super().forward(
            x,
            timesteps,
            context,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            **kwargs,
        )
