# Copyright 2026 Aozora Trainer contributors
# Licensed under the Apache License, Version 2.0. See the repository LICENSE.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import torch
from diffusers import AutoencoderKL, UNet2DConditionModel
from safetensors import safe_open
from safetensors.torch import load_file


def load_sdxl_unet(path, compute_dtype):
    """Load an SDXL UNet while detecting nonstandard input/output channels."""
    print(f"INFO: Loading UNet from: {Path(path).name}")
    in_channels = 4
    out_channels = 4
    try:
        with safe_open(path, framework="pt", device="cpu") as handle:
            key_in = "model.diffusion_model.input_blocks.0.0.weight"
            key_out = "model.diffusion_model.out.2.weight"
            if key_in in handle.keys():
                shape_in = handle.get_slice(key_in).get_shape()
                in_channels = shape_in[1]
            if key_out in handle.keys():
                shape_out = handle.get_slice(key_out).get_shape()
                out_channels = shape_out[0]
    except Exception as e:
        print(
            "WARNING: Could not peek into safetensors for channel sizes, "
            f"falling back to defaults. Error: {e}"
        )

    print(
        "INFO: Detected UNet configuration - "
        f"in_channels: {in_channels}, out_channels: {out_channels}"
    )

    try:
        unet = UNet2DConditionModel.from_single_file(
            path,
            torch_dtype=compute_dtype,
            low_cpu_mem_usage=True,
            in_channels=in_channels,
            out_channels=out_channels,
        )
        print(
            "INFO: Loaded UNet "
            f"(Channels: In={unet.config.in_channels} / Out={unet.config.out_channels})"
        )
        return unet
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load UNet. {e}")
        raise


def load_sdxl_vae(path, device, target_channels=None):
    """Load an SDXL-compatible VAE and detect its latent channel count."""
    print(f"INFO: Attempting to load VAE from: {path}")
    if target_channels is None:
        try:
            tensors = load_file(path, device="cpu")
            for key in ("first_stage_model.quant_conv.weight", "quant_conv.weight"):
                if key in tensors:
                    target_channels = tensors[key].shape[0] // 2
                    break
        except Exception:
            pass

    target_channels = target_channels or 4
    try:
        if target_channels != 4:
            vae = AutoencoderKL.from_single_file(
                path,
                torch_dtype=torch.float32,
                latent_channels=target_channels,
                ignore_mismatched_sizes=True,
                low_cpu_mem_usage=False,
            )
        else:
            vae = AutoencoderKL.from_single_file(
                path,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
            )
        print(f"INFO: Successfully loaded VAE with {target_channels} channels.")
        return vae.to(device)
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load VAE: {e}")
        raise
