# Copyright 2026 Aozora Trainer contributors
# Licensed under the Apache License, Version 2.0. See the repository LICENSE.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from diffusers import StableDiffusionXLPipeline


class SDXLTrainingComponents:
    """Minimal caching-time container for the SDXL components Aozora uses."""

    def __init__(self):
        self.tokenizer = None
        self.tokenizer_2 = None
        self.text_encoder = None
        self.text_encoder_2 = None
        self.vae = None

    @classmethod
    def from_single_file(
        cls,
        checkpoint_path,
        vae,
        torch_dtype=torch.float32,
        **pipeline_kwargs,
    ):
        pipeline = StableDiffusionXLPipeline.from_single_file(
            checkpoint_path,
            vae=vae,
            unet=None,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            **pipeline_kwargs,
        )
        components = cls()
        components.tokenizer = pipeline.tokenizer
        components.tokenizer_2 = pipeline.tokenizer_2
        components.text_encoder = pipeline.text_encoder
        components.text_encoder_2 = pipeline.text_encoder_2
        components.vae = pipeline.vae
        return components
