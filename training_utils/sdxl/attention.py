# Copyright 2026 Aozora Trainer contributors
# Licensed under the Apache License, Version 2.0. See the repository LICENSE.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import AttnProcessor2_0

try:
    from flash_attn import flash_attn_func
except ImportError:
    flash_attn_func = None


class FlashAttnProcessor2_0:
    """Diffusers attention processor backed directly by Flash Attention 2."""

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        *args,
        **kwargs,
    ):
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim)
        key = key.view(batch_size, -1, attn.heads, head_dim)
        value = value.view(batch_size, -1, attn.heads, head_dim)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        can_use_flash_attn = (
            attention_mask is None
            and query.is_cuda
            and query.dtype in (torch.float16, torch.bfloat16)
        )
        if can_use_flash_attn:
            hidden_states = flash_attn_func(query, key, value)
        else:
            hidden_states = F.scaled_dot_product_attention(
                query.transpose(1, 2),
                key.transpose(1, 2),
                value.transpose(1, 2),
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
            ).transpose(1, 2)

        hidden_states = hidden_states.reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        return hidden_states / attn.rescale_output_factor


def set_attention_processor(unet, attention_mode="flash_attn"):
    """Configure the requested attention backend on an SDXL Diffusers UNet."""
    if attention_mode == "cudnn":
        if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
            torch.backends.cuda.enable_cudnn_sdp(True)
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            unet.set_attn_processor(AttnProcessor2_0())
            print("INFO: Using CuDNN SDPA backend (PyTorch 2.5+ optimized)")
        else:
            print("WARNING: CuDNN SDPA requires PyTorch 2.5+, falling back to standard SDPA")
            unet.set_attn_processor(AttnProcessor2_0())
    elif attention_mode in ("xformers", "xformers (Only if no Flash)"):
        unet.enable_xformers_memory_efficient_attention()
        print("INFO: Using xFormers")
    elif attention_mode in ("flash_attn", "flash_attn (Flash Attention 2)"):
        if flash_attn_func is None:
            raise ImportError(
                "Flash Attention 2 was selected, but the flash-attn package could not be imported. "
                "Install it in portable_Venv or rerun setup.bat and choose Flash Attention."
            )
        unet.set_attn_processor(FlashAttnProcessor2_0())
        print("INFO: Using Flash Attention 2 package directly")
    elif attention_mode == "pytorch29_optimized":
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)
            unet.set_attn_processor(AttnProcessor2_0())
            print("INFO: Using PyTorch 2.9 Optimized SDPA (Flash + MemEfficient + Math)")
            print(f"      CUDA Version: {torch.version.cuda}")
            print(f"      PyTorch Version: {torch.__version__}")
        except Exception as e:
            print(f"WARNING: PyTorch 2.9 optimization failed: {e}")
            unet.set_attn_processor(AttnProcessor2_0())
    else:
        unet.set_attn_processor(AttnProcessor2_0())
        print("INFO: Using SDPA (PyTorch native)")
