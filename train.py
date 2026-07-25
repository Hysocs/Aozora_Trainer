import inspect
import fnmatch
import os
import hashlib
from pathlib import Path
import gc
import json
from collections import defaultdict, deque
import random
import time
import math
import re
import secrets
import string
import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader, Sampler
from diffusers import StableDiffusionXLPipeline, AutoencoderKL, FlowMatchEulerDiscreteScheduler, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from safetensors import safe_open
from safetensors.torch import save_file, load_file
from PIL import Image, TiffImagePlugin, ImageFile, ImageOps
from torchvision import transforms
from tqdm.auto import tqdm
import logging
import warnings
import argparse
import numpy as np
import cv2
import multiprocessing
from multiprocessing import Pool, cpu_count
from training_utils.config import config as default_config
import threading
import queue
from training_utils.optimizers.raven import RavenAdamW
from training_utils.optimizers.titan import TitanAdamW
from diffusers.models.attention_processor import (
    AttnProcessor2_0,
    FusedAttnProcessor2_0
)
from training_utils.caching.cache import (
    CAPTION_JSON_PRIMARY_TYPE,
    CAPTION_JSON_TYPES,
    cache_base_stem_from_cache_path,
    cache_base_stem_from_te_path,
    cache_index_exists,
    cache_item_stem_from_te_path,
    cache_metadata_matches,
    cache_image_layout_options_match,
    cache_latent_options_match,
    cache_stem_for_image,
    cache_text_options_match,
    cached_file_signatures_match,
    caption_file_signature_for_image,
    caption_source_type,
    caption_types_for_cache,
    caption_variant_index,
    collect_image_paths,
    expected_cache_paths_for_metadata,
    image_file_signature,
    json_caption_cache_suffix,
    json_caption_mode_enabled,
    load_cache_index,
    remove_cache_files_for_stem,
    remove_cache_pair_for_te_path,
    save_cache_index,
    selected_caption_variant_path,
    stable_cache_item_key,
    strip_json_caption_suffix,
    te_paths_for_index_item,
    text_cache_paths_for_index_item,
)

warnings.filterwarnings("ignore", category=UserWarning, module=TiffImagePlugin.__name__, message="Corrupt EXIF data")
Image.MAX_IMAGE_PIXELS = 190_000_000
ImageFile.LOAD_TRUNCATED_IMAGES = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
os.environ["TOKENIZERS_PARALLELISM"] = "false"

FLUX_BN_EPS = 1e-4


def get_json_caption_weights(config):
    weights = {
        "tags": int(getattr(config, "CAPTION_TAGS_PERCENT", 40) or 0),
        "nl": int(getattr(config, "CAPTION_NL_PERCENT", 10) or 0),
        "tags_nl": int(getattr(config, "CAPTION_TAGS_NL_PERCENT", 25) or 0),
        "nl_tags": int(getattr(config, "CAPTION_NL_TAGS_PERCENT", 25) or 0),
    }
    weights = {k: max(0, v) for k, v in weights.items()}
    if sum(weights.values()) <= 0:
        weights[CAPTION_JSON_PRIMARY_TYPE] = 100
    return weights


def sdxl_text_cache_compatible_options(cached_options, expected_options):
    return cache_text_options_match(cached_options, expected_options)


def sdxl_lat_cache_compatible_options(cached_options, expected_options):
    return cache_latent_options_match(cached_options, expected_options)


def sdxl_text_cache_valid(path, root, meta, caption_type, caption, text_cache_dtype, expected_options):
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        embeds = payload.get("embeds")
        pooled = payload.get("pooled")
        return (
            embeds is not None
            and pooled is not None
            and embeds.dtype == text_cache_dtype
            and pooled.dtype == text_cache_dtype
            and payload.get("caption_type") == caption_type
            and payload.get("caption") == caption
            and payload.get("caption_signature") == meta.get("caption_signature")
            and cache_metadata_matches(payload, root, meta)
            and sdxl_text_cache_compatible_options(payload.get("cache_options"), expected_options)
        )
    except Exception:
        return False


def sdxl_lat_cache_valid(path, root, meta, vae_cache_dtype, expected_options):
    try:
        lat_payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(lat_payload, dict):
            return False
        if not cache_metadata_matches(lat_payload, root, meta):
            return False
        if not sdxl_lat_cache_compatible_options(lat_payload.get("cache_options"), expected_options):
            return False
        latents = lat_payload.get("latents")
        return (
            latents is not None
            and latents.dtype == vae_cache_dtype
            and not torch.isnan(latents).any()
            and not torch.isinf(latents).any()
        )
    except Exception:
        return False
CLIP_CHUNK_TOKEN_COUNT = 77


def cache_float_dtype(config, attr_name):
    precision = str(getattr(config, attr_name, "bfloat16") or "bfloat16").strip().lower()
    aliases = {
        "fp32": "float32",
        "float": "float32",
        "bf16": "bfloat16",
        "bfp16": "bfloat16",
        "fp16": "float16",
        "half": "float16",
    }
    precision = aliases.get(precision, precision)
    if precision == "float32":
        return torch.float32
    if precision == "float16":
        return torch.float16
    return torch.bfloat16


def cache_float_dtype_name(config, attr_name):
    return str(cache_float_dtype(config, attr_name)).replace("torch.", "")


def text_cache_float_dtype(config):
    return cache_float_dtype(config, "TEXT_CACHE_PRECISION")


def text_cache_float_dtype_name(config):
    return cache_float_dtype_name(config, "TEXT_CACHE_PRECISION")


def vae_cache_float_dtype(config):
    return cache_float_dtype(config, "VAE_CACHE_PRECISION")


def vae_cache_float_dtype_name(config):
    return cache_float_dtype_name(config, "VAE_CACHE_PRECISION")

BN_MEAN_SUFFIXES = [
    "bn.running_mean",
    "normalize.bn.running_mean",
    "normalize.running_mean",
]
BN_VAR_SUFFIXES = [
    "bn.running_var",
    "normalize.bn.running_var",
    "normalize.running_var",
]


def set_attention_processor(unet, attention_mode="flash_attn"):

    if attention_mode == "cudnn":
        if hasattr(torch.backends.cuda, 'enable_cudnn_sdp'):
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
    elif attention_mode == "flash_attn":
        if hasattr(torch.backends.cuda, 'enable_cudnn_sdp'):
            torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(False)
        unet.set_attn_processor(AttnProcessor2_0())
        print("INFO: Using Flash Attention SDPA backend")
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

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    print(f"INFO: Set random seed to {seed}")

def fix_alpha_channel(img):
    if img.mode == 'P' and 'transparency' in img.info:
        img = img.convert('RGBA')
    if img.mode in ('RGBA', 'PA', 'LA'):
        rgb_img = img.convert('RGB')
        return rgb_img
    return img.convert("RGB")

def generate_noise(latents, generator, device, dtype=None, step=None, seed=None):
    # Optional deterministic step-seeding
    if step is not None and seed is not None:
        step_seed = (seed + step) % (2**32 - 1)
        generator.manual_seed(step_seed)

    return torch.randn(latents.shape, device=device, dtype=torch.float32, generator=generator)


def seeded_torch_generator(device, seed, *parts):
    value = int(seed if seed else 42) & 0xFFFFFFFFFFFFFFFF
    for part in parts:
        value = (value * 6364136223846793005 + int(part) + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
    generator = torch.Generator(device=device)
    generator.manual_seed(value % (2**63 - 1))
    return generator


class TrainingConfig:
    def __init__(self):
        for key, value in default_config.flat_defaults().items():
            setattr(self, key, value)
        self._load_from_user_config()
        self._type_check_and_correct()
        self.NOISE_MODE = "normal"
        self.compute_dtype = torch.bfloat16 if self.MIXED_PRECISION == "bfloat16" else torch.float16
        self.is_rectified_flow = getattr(self, "PREDICTION_TYPE", "epsilon") == "rectified_flow"

    def _load_from_user_config(self):
        parser = argparse.ArgumentParser(description="Load a specific training configuration.")
        parser.add_argument("--config", type=str, help="Path to the user configuration JSON file.")
        args, _ = parser.parse_known_args()
        if args.config:
            path = Path(args.config)
            if path.exists():
                print(f"INFO: Loading configuration from {path}")
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        user_config = default_config.flatten_preset(json.load(f))
                    for key, value in user_config.items():
                        setattr(self, key, value)
                except (json.JSONDecodeError, TypeError) as e:
                    print(f"ERROR: Could not parse {path}: {e}. Using defaults.")
            else:
                print(f"WARNING: Config {path} not found. Using defaults.")

    def _type_check_and_correct(self):
        if getattr(self, "RESUME_TRAINING", False):
            is_anima = str(getattr(self, "TRAINING_MODE", "")).lower().startswith("anima")
            resume_path_keys = (
                ["ANIMA_RESUME_MODEL_PATH", "ANIMA_RESUME_STATE_PATH"]
                if is_anima
                else ["RESUME_MODEL_PATH", "RESUME_STATE_PATH"]
            )
            for key in resume_path_keys:
                value = getattr(self, key, "")
                if not value or not Path(value).exists():
                    raise FileNotFoundError(f"RESUME_TRAINING is enabled, but {key}='{value}' is not a valid file path.")

        for key, value in list(self.__dict__.items()):
            if key == "UNET_EXCLUDE_TARGETS":
                if isinstance(value, str): setattr(self, key, [item.strip() for item in value.split(',') if item.strip()])
                elif isinstance(value, list): setattr(self, key, [item for item in value if item])
                continue
            default_value = getattr(default_config, key, None)
            if default_value is None or isinstance(value, type(default_value)): continue
            expected_type = type(default_value)
            if expected_type == bool and isinstance(value, str):
                setattr(self, key, value.lower() in ['true', '1', 't', 'y', 'yes'])
                continue
            try:
                if expected_type == int: setattr(self, key, int(float(value)))
                else: setattr(self, key, expected_type(value))
            except (ValueError, TypeError):
                setattr(self, key, default_value)


class CustomCurveLRScheduler:
    def __init__(self, optimizer, curve_points, total_micro_steps):
        self.optimizer = optimizer
        self.curve_points = sorted(curve_points, key=lambda p: p[0])
        self.total_micro_steps = max(total_micro_steps, 1)
        self.current_micro_step = 0
        if not self.curve_points: raise ValueError("LR_CUSTOM_CURVE cannot be empty")
        if self.curve_points[0][0] != 0.0: self.curve_points.insert(0, [0.0, self.curve_points[0][1]])
        if self.curve_points[-1][0] != 1.0: self.curve_points.append([1.0, self.curve_points[-1][1]])
        self._update_lr()

    def _interpolate_lr(self, normalized_position):
        normalized_position = max(0.0, min(1.0, normalized_position))
        for i in range(len(self.curve_points) - 1):
            x1, y1 = self.curve_points[i]
            x2, y2 = self.curve_points[i + 1]
            if x1 <= normalized_position <= x2:
                if x2 - x1 == 0: return y1
                t = (normalized_position - x1) / (x2 - x1)
                return y1 + t * (y2 - y1)
        return self.curve_points[-1][1]

    def _update_lr(self):
        normalized_position = self.current_micro_step / max(self.total_micro_steps - 1, 1)
        lr = self._interpolate_lr(normalized_position)
        for param_group in self.optimizer.param_groups:
            scale = param_group.get('lr_scale', 1.0)
            param_group['lr'] = lr * scale

    def step(self, micro_step):
        self.current_micro_step = micro_step
        self._update_lr()

    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]


class TrainingDiagnostics:
    def __init__(self, accumulation_steps):
        self.accumulation_steps = accumulation_steps
        self.losses = deque(maxlen=accumulation_steps)

    def step(self, loss):
        if loss is not None: self.losses.append(loss)

    def get_average_loss(self):
        if not self.losses: return 0.0
        return sum(self.losses) / len(self.losses)

    def reset(self):
        self.losses.clear()


# =========================================================================
# ASYNC REPORTER: 100% OFFLOADED LOGGING
# =========================================================================
class AsyncReporter:
    def __init__(self, total_steps, test_param_name):
        self.total_steps = total_steps
        self.test_param_name = test_param_name
        self.task_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
        self._last_line_len = 0

    def _clear_line(self):
        if self._last_line_len > 0:
            print('\r' + ' ' * self._last_line_len + '\r', end='', flush=True)
            self._last_line_len = 0

    def _format_time(self, seconds):
        if seconds is None or not math.isfinite(seconds): return "N/A"
        seconds = int(seconds)
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{hours:02}:{minutes:02}:{secs:02}"

    def _worker(self):
        while not self.stop_event.is_set():
            try:
                task_type, data = self.task_queue.get(timeout=0.05)
                if task_type == 'log_step': self._handle_log_step(**data)
                elif task_type == 'message': self._handle_message(**data)
                self.task_queue.task_done()
            except queue.Empty: continue

    def _handle_log_step(self, global_step, timing_data, diag_data):
        if diag_data:
            self._clear_line()
            update_status = "[OK]" if diag_data['update_delta'] > 1e-12 else "[NO UPDATE!]"
            print(
                f"\n--- Optimizer Step: {diag_data['optim_step']:<5} | Loss: {diag_data['avg_loss']:<8.5f} | LR: {diag_data['current_lr']:.2e} ---\n"
                f"  Time: {diag_data['optim_step_time']:.2f}s/step | Avg Speed: {diag_data['avg_optim_step_time']:.2f}s/step\n"
                f"  Grad Norm (Raw/Clipped): {diag_data['raw_grad_norm']:<8.4f} / {diag_data['clipped_grad_norm']:<8.4f}\n"
                f"  VRAM: Training={torch.cuda.memory_reserved()/1e9:.2f}GB | Model={torch.cuda.memory_allocated()/1e9:.2f}GB\n"
                f"  |- Update Magnitude : {diag_data['update_delta']:.4e} {update_status}\n"
            )

        bar_width = 30
        percentage = (global_step + 1) / self.total_steps
        filled_length = int(bar_width * percentage)
        bar = '#' * filled_length + '-' * (bar_width - filled_length)
        s_per_step = timing_data.get('raw_step_time', 0)
        time_spent_str = self._format_time(timing_data.get('elapsed_time'))
        eta_str = self._format_time(timing_data.get('eta'))
        loss_val = timing_data.get('loss', 0.0)
        timestep_val = timing_data.get('timestep', 'N/A')
        sigma_val = timing_data.get('sigma')
        sampling_text = (
            f"Ticket: {timestep_val}, Sigma: {float(sigma_val):.6f}"
            if sigma_val is not None else f"Timestep: {timestep_val}"
        )
        prog_str = f'Training |{bar}| {global_step + 1}/{self.total_steps}[{percentage:.2%}][Loss: {loss_val:.4f}, {sampling_text}][{s_per_step:.2f}s/step, ETA: {eta_str}, Elapsed: {time_spent_str}]'
        print('\r' + prog_str, end='', flush=True)
        self._last_line_len = len(prog_str)

    def _handle_message(self, text):
        self._clear_line()
        print(text)

    def log_step(self, global_step, timing_data, diag_data=None):
        self.task_queue.put(('log_step', {'global_step': global_step, 'timing_data': timing_data, 'diag_data': diag_data}))

    def log_message(self, text):
        self.task_queue.put(('message', {'text': text}))

    def shutdown(self):
        self._clear_line()
        print("\nShutting down async reporter. Waiting for pending tasks...")
        self.task_queue.join()
        self.stop_event.set()
        self.worker_thread.join()


class BucketBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, seed, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0
        self.start_batch_index = 0
        self.total_images = len(self.dataset)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def set_start_batch_index(self, batch_index):
        self.start_batch_index = max(0, int(batch_index or 0))

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        if self.batch_size == 1:
            indices = torch.randperm(self.total_images, generator=g).tolist()
            batches = [[i] for i in indices]
        else:
            indices = torch.randperm(self.total_images, generator=g).tolist()
            buckets = defaultdict(list)
            for idx in indices:
                buckets[self.dataset.bucket_keys[idx]].append(idx)

            bucket_batches = {}
            for key in sorted(buckets):
                bucket_indices = buckets[key]
                chunks = [
                    bucket_indices[i:i + self.batch_size]
                    for i in range(0, len(bucket_indices), self.batch_size)
                ]
                if self.shuffle and len(chunks) > 1:
                    chunk_order = torch.randperm(len(chunks), generator=g).tolist()
                    chunks = [chunks[i] for i in chunk_order]
                bucket_batches[key] = chunks

            if self.shuffle:
                batches = []
                last_key = None
                while bucket_batches:
                    candidates = [key for key in bucket_batches if key != last_key]
                    if not candidates:
                        candidates = list(bucket_batches)

                    max_remaining = max(len(bucket_batches[key]) for key in candidates)
                    top_candidates = [
                        key for key in candidates
                        if len(bucket_batches[key]) == max_remaining
                    ]
                    key_index = torch.randint(len(top_candidates), (1,), generator=g).item()
                    key = top_candidates[key_index]

                    batches.append(bucket_batches[key].pop(0))
                    last_key = key
                    if not bucket_batches[key]:
                        del bucket_batches[key]
            else:
                batches = [
                    batch
                    for key in sorted(bucket_batches)
                    for batch in bucket_batches[key]
                ]

        if self.start_batch_index > 0:
            batches = batches[self.start_batch_index:]
            self.start_batch_index = 0

        self.epoch += 1
        yield from batches

    def __len__(self):
        return math.ceil(self.total_images / self.batch_size)


class PrecomputedImageBatchSampler(Sampler):
    def __init__(self, image_batches, seed, start_step=0):
        self.image_batches = image_batches
        self.seed = seed
        self.start_step = max(0, int(start_step or 0))
        self.epoch = 0

    def __iter__(self):
        for step in range(self.start_step, len(self.image_batches)):
            self.epoch = step + 1
            batch = self.image_batches[step]
            if isinstance(batch, np.ndarray):
                yield [int(index) for index in batch.tolist()]
            else:
                yield [int(index) for index in batch]

    def __len__(self):
        return max(0, len(self.image_batches) - self.start_step)

    def set_epoch(self, epoch):
        self.epoch = int(epoch or 0)

    def set_start_batch_index(self, batch_index):
        self.start_step = max(0, int(batch_index or 0))


def timestep_bin_ids(timesteps, bin_ranges):
    bin_ids = np.zeros(len(timesteps), dtype=np.int32)
    for step, timestep in enumerate(timesteps):
        timestep = int(timestep)
        for bin_id, (start_t, end_t) in enumerate(bin_ranges):
            if start_t <= timestep < end_t:
                bin_ids[step] = bin_id
                break
    return bin_ids


def _scale_timestep_counts(counts, target_total):
    target_total = max(0, int(target_total))
    counts = [max(0, int(count or 0)) for count in counts]
    total = sum(counts)
    if target_total <= 0 or total <= 0:
        return [0 for _ in counts]
    raw = [(count / total) * target_total for count in counts]
    scaled = [int(value) for value in raw]
    diff = target_total - sum(scaled)
    if diff > 0:
        fractions = sorted(
            range(len(raw)),
            key=lambda index: raw[index] - scaled[index],
            reverse=True,
        )
        for index in fractions[:diff]:
            scaled[index] += 1
    return scaled


def _build_timestep_bin_counts(allocation, total_tickets_needed, total_timestep_count):
    if not allocation or "counts" not in allocation or "bin_size" not in allocation or sum(allocation["counts"]) == 0:
        bin_size = max(1, int(1000 / 10))
        bins = max(1, math.ceil(1000 / bin_size))
        counts = [total_tickets_needed // bins] * bins
        for index in range(total_tickets_needed % bins):
            counts[index] += 1
    else:
        bin_size = max(1, int(allocation["bin_size"]))
        counts = _scale_timestep_counts(allocation["counts"], total_tickets_needed)

    scale_factor = total_timestep_count / 1000.0
    bin_counts = []
    bin_ranges = []
    for index, count in enumerate(counts):
        if count <= 0:
            continue
        start_t = int(index * bin_size * scale_factor)
        end_t = min(total_timestep_count, max(start_t + 1, int((index + 1) * bin_size * scale_factor)))
        if start_t >= total_timestep_count:
            break
        bin_counts.append(int(count))
        bin_ranges.append((start_t, end_t))
    return bin_counts, bin_ranges


def _build_balanced_timestep_bin_order(bin_counts, seed):
    if not bin_counts:
        return []
    rng = np.random.Generator(np.random.PCG64(seed + 7919))
    positions = []
    bins = []
    jitter = []
    for bin_id, count in enumerate(bin_counts):
        if count <= 0:
            continue
        positions.append((np.arange(count, dtype=np.float64) + rng.random(count)) / count)
        bins.append(np.full(count, bin_id, dtype=np.int32))
        jitter.append(rng.random(count))
    if not positions:
        return []
    all_positions = np.concatenate(positions)
    all_bins = np.concatenate(bins)
    all_jitter = np.concatenate(jitter)
    order = np.lexsort((all_jitter, all_positions))
    return all_bins[order].tolist()


def _build_stratified_timestep_pool(bin_counts, bin_ranges, seed):
    rng = np.random.Generator(np.random.PCG64(seed))
    value_decks = []
    for count, (start_t, end_t) in zip(bin_counts, bin_ranges):
        values = np.arange(start_t, end_t, dtype=np.int64)
        deck = []
        while len(deck) < count:
            shuffled = rng.permutation(values).tolist()
            deck.extend(shuffled[: count - len(deck)])
        value_decks.append(deck)

    positions = [0 for _ in value_decks]
    pool = []
    for bin_id in _build_balanced_timestep_bin_order(bin_counts, seed):
        pos = positions[bin_id]
        pool.append(int(value_decks[bin_id][pos]))
        positions[bin_id] = pos + 1
    return pool


def build_timestep_ticket_pool(allocation, total_tickets_needed, total_timestep_count=1000, seed=42, stratified=False):
    total_tickets_needed = max(0, int(total_tickets_needed))
    total_timestep_count = max(1, int(total_timestep_count))
    seed = int(seed if seed else 42)
    bin_counts, bin_ranges = _build_timestep_bin_counts(allocation, total_tickets_needed, total_timestep_count)

    if stratified:
        pool = _build_stratified_timestep_pool(bin_counts, bin_ranges, seed)
    else:
        rng = np.random.Generator(np.random.PCG64(seed))
        pool = []
        for count, (start_t, end_t) in zip(bin_counts, bin_ranges):
            pool.extend(rng.integers(start_t, end_t, size=max(1, int(count))).tolist())
        random.Random(seed).shuffle(pool)

    if not pool:
        fallback_rng = random.Random(seed)
        pool = [fallback_rng.randint(0, total_timestep_count - 1) for _ in range(total_tickets_needed)]
    while len(pool) < total_tickets_needed:
        pool.extend(pool[: total_tickets_needed - len(pool)])
    return pool[:total_tickets_needed], bin_ranges


def build_epoch_shuffle_image_schedule(total_images, total_steps, seed):
    schedule = np.empty(total_steps, dtype=np.uint32)
    offset = 0
    epoch = 0
    while offset < total_steps:
        g = torch.Generator()
        g.manual_seed(seed + epoch)
        order = torch.randperm(total_images, generator=g).numpy().astype(np.uint32, copy=False)
        take = min(total_images, total_steps - offset)
        schedule[offset:offset + take] = order[:take]
        offset += take
        epoch += 1
    return schedule


def build_spread_image_schedule(total_images, total_steps, seed, bin_ids, bin_count):
    if total_images <= 0 or total_steps <= 0:
        return np.empty(0, dtype=np.uint32)
    if bin_count <= 1:
        return build_epoch_shuffle_image_schedule(total_images, total_steps, seed)

    history_depth = max(1, min(bin_count, math.ceil(total_steps / total_images)))
    sentinel = 255 if bin_count < 255 else 65535
    history_dtype = np.uint8 if bin_count < 255 else np.uint16
    recent_bins = np.full((total_images, history_depth), sentinel, dtype=history_dtype)
    recent_pos = np.zeros(total_images, dtype=np.uint16)
    schedule = np.empty(total_steps, dtype=np.uint32)
    offset = 0
    epoch = 0

    while offset < total_steps:
        epoch_steps = min(total_images, total_steps - offset)
        remaining = np.ones(total_images, dtype=np.bool_)
        queues = {}
        positions = {}
        rng = np.random.Generator(np.random.PCG64(seed + 104729 + epoch))

        for local_step in range(epoch_steps):
            step = offset + local_step
            bin_id = int(bin_ids[step])
            queue = queues.get(bin_id)
            if queue is None:
                queue = rng.permutation(total_images).astype(np.uint32, copy=False)
                queues[bin_id] = queue
                positions[bin_id] = 0

            chosen = None
            pos = positions[bin_id]
            while pos < total_images:
                candidate = int(queue[pos])
                pos += 1
                if remaining[candidate] and not np.any(recent_bins[candidate] == bin_id):
                    chosen = candidate
                    break
            positions[bin_id] = pos

            if chosen is None:
                remaining_indices = np.flatnonzero(remaining)
                if remaining_indices.size == 0:
                    break
                penalties = np.count_nonzero(recent_bins[remaining_indices] == bin_id, axis=1)
                best_penalty = penalties.min()
                best_indices = remaining_indices[penalties == best_penalty]
                chosen = int(best_indices[int(rng.integers(0, len(best_indices)))])

            schedule[step] = chosen
            remaining[chosen] = False
            pos_idx = int(recent_pos[chosen] % history_depth)
            recent_bins[chosen, pos_idx] = bin_id
            recent_pos[chosen] = (recent_pos[chosen] + 1) % history_depth

        offset += epoch_steps
        epoch += 1

    return schedule


def build_image_schedule(total_images, total_steps, seed, timesteps, bin_ranges, force_spread):
    if not force_spread:
        return build_epoch_shuffle_image_schedule(total_images, total_steps, seed)
    return build_spread_image_schedule(
        total_images,
        total_steps,
        seed,
        timestep_bin_ids(timesteps, bin_ranges),
        len(bin_ranges),
    )


def build_epoch_shuffle_batch_schedule(dataset, total_steps, batch_size, seed):
    schedule = []
    epoch = 0
    while len(schedule) < total_steps:
        sampler = BucketBatchSampler(dataset, batch_size, seed, shuffle=True)
        sampler.set_epoch(epoch)
        for batch in sampler:
            schedule.append([int(index) for index in batch])
            if len(schedule) >= total_steps:
                break
        epoch += 1
    return schedule


def build_spread_batch_schedule(dataset, total_steps, batch_size, seed, timesteps, bin_ranges):
    total_images = len(dataset)
    if total_images <= 0 or total_steps <= 0:
        return []
    if batch_size == 1:
        image_schedule = build_image_schedule(total_images, total_steps, seed, timesteps, bin_ranges, True)
        return [[int(index)] for index in image_schedule.tolist()]

    bin_ids = timestep_bin_ids(timesteps, bin_ranges)
    total_samples = min(len(timesteps), total_steps * batch_size)
    bin_count = max(1, len(bin_ranges))
    history_depth = max(1, min(bin_count, math.ceil(total_samples / total_images)))
    sentinel = 255 if bin_count < 255 else 65535
    history_dtype = np.uint8 if bin_count < 255 else np.uint16
    recent_bins = np.full((total_images, history_depth), sentinel, dtype=history_dtype)
    recent_pos = np.zeros(total_images, dtype=np.uint16)
    bucket_indices = defaultdict(list)
    for index, key in enumerate(dataset.bucket_keys):
        bucket_indices[key].append(index)

    schedule = []
    sample_offset = 0
    epoch = 0
    while len(schedule) < total_steps:
        base_sampler = BucketBatchSampler(dataset, batch_size, seed, shuffle=True)
        base_sampler.set_epoch(epoch)
        remaining = np.ones(total_images, dtype=np.bool_)
        queues = {}
        positions = {}
        rng = np.random.Generator(np.random.PCG64(seed + 104729 + epoch))

        for base_batch in base_sampler:
            if len(schedule) >= total_steps:
                break
            batch_len = len(base_batch)
            bucket_key = dataset.bucket_keys[base_batch[0]]
            chosen_batch = []

            for local_index in range(batch_len):
                if sample_offset + local_index >= len(bin_ids):
                    break
                bin_id = int(bin_ids[sample_offset + local_index])
                queue_key = (bucket_key, bin_id)
                queue = queues.get(queue_key)
                if queue is None:
                    queue = np.array(bucket_indices[bucket_key], dtype=np.uint32)
                    rng.shuffle(queue)
                    queues[queue_key] = queue
                    positions[queue_key] = 0

                chosen = None
                pos = positions[queue_key]
                while pos < len(queue):
                    candidate = int(queue[pos])
                    pos += 1
                    if remaining[candidate] and not np.any(recent_bins[candidate] == bin_id):
                        chosen = candidate
                        break
                positions[queue_key] = pos

                if chosen is None:
                    remaining_indices = np.array(
                        [index for index in bucket_indices[bucket_key] if remaining[index]],
                        dtype=np.int64,
                    )
                    if remaining_indices.size == 0:
                        break
                    penalties = np.count_nonzero(recent_bins[remaining_indices] == bin_id, axis=1)
                    best_penalty = penalties.min()
                    best_indices = remaining_indices[penalties == best_penalty]
                    chosen = int(best_indices[int(rng.integers(0, len(best_indices)))])

                chosen_batch.append(chosen)
                remaining[chosen] = False
                pos_idx = int(recent_pos[chosen] % history_depth)
                recent_bins[chosen, pos_idx] = bin_id
                recent_pos[chosen] = (recent_pos[chosen] + 1) % history_depth

            if chosen_batch:
                schedule.append(chosen_batch)
                sample_offset += len(chosen_batch)
            if sample_offset >= len(bin_ids):
                break

        epoch += 1
    return schedule


def build_image_batch_schedule(dataset, total_steps, batch_size, seed, timesteps, bin_ranges, force_spread):
    if not force_spread:
        return build_epoch_shuffle_batch_schedule(dataset, total_steps, batch_size, seed)
    return build_spread_batch_schedule(dataset, total_steps, batch_size, seed, timesteps, bin_ranges)


STANDARD_SDXL_BUCKETS = [
    (1024, 1024),
    (1152, 896), (896, 1152),
    (1216, 832), (832, 1216),
    (1344, 768), (768, 1344),
    (1440, 720), (720, 1440),
    (1536, 640), (640, 1536),
    (1600, 512), (512, 1600),
    (896, 896), (768, 768),
]
LOW_RES_ASPECT_BUCKETS = [
    (1152, 512), (512, 1152),
    (1024, 576), (576, 1024),
    (960, 640), (640, 960),
    (896, 704), (704, 896),
    (768, 768),
]
MAX_BUCKET_RESOLUTION_CHOICES = (896, 1024, 1152, 1536)
BUCKET_LAYOUT_VERSION = "preset_ladder_v3"


def resolve_max_bucket_resolution(value=None):
    if value is None:
        return 1024
    try:
        numeric = int(float(value))
    except (TypeError, ValueError):
        return 1024

    if numeric > 4096:
        numeric = int(round(math.sqrt(max(1, numeric))))

    valid = [size for size in MAX_BUCKET_RESOLUTION_CHOICES if size <= numeric]
    return valid[-1] if valid else MAX_BUCKET_RESOLUTION_CHOICES[0]


def get_max_bucket_resolution_for_config(config):
    if hasattr(config, "MAX_BUCKET_RESOLUTION"):
        return resolve_max_bucket_resolution(getattr(config, "MAX_BUCKET_RESOLUTION"))
    return resolve_max_bucket_resolution(default_config.MAX_BUCKET_RESOLUTION)


def get_bucket_ladder(max_bucket_resolution=None):
    max_bucket_resolution = resolve_max_bucket_resolution(max_bucket_resolution)
    buckets = set()
    if max_bucket_resolution < 1024:
        tiers = [max_bucket_resolution]
    else:
        tiers = [1024, *[tier for tier in (1152, 1536) if tier <= max_bucket_resolution]]

    for tier in tiers:
        if tier == 1024:
            buckets.update(STANDARD_SDXL_BUCKETS)
            buckets.update(LOW_RES_ASPECT_BUCKETS)
            continue
        scale = tier / 1024
        for width, height in STANDARD_SDXL_BUCKETS + LOW_RES_ASPECT_BUCKETS:
            scaled_w = max(64, int(round((width * scale) / 64)) * 64)
            scaled_h = max(64, int(round((height * scale) / 64)) * 64)
            buckets.add((scaled_w, scaled_h))

    return sorted(buckets, key=lambda item: (item[0] * item[1], item[0], item[1]))


def get_optimal_bucket(orig_w, orig_h, target_area=None, stride=64, should_upscale=False):
    orig_ar = orig_w / max(orig_h, 1)
    max_bucket_resolution = resolve_max_bucket_resolution(target_area)
    candidate_buckets = get_bucket_ladder(max_bucket_resolution)
    target_area = max_bucket_resolution * max_bucket_resolution

    def bucket_score(bw, bh):
        bucket_ar = bw / max(bh, 1)
        bucket_area = bw * bh
        ar_error = abs(bucket_ar - orig_ar) / max(orig_ar, 0.01)
        area_error = abs(math.log(bucket_area / target_area)) if bucket_area > 0 else 100.0
        return ar_error * 10.0 + area_error

    best_bucket = min(candidate_buckets, key=lambda b: bucket_score(b[0], b[1]))
    bw, bh = best_bucket

    if not should_upscale and (bw > orig_w or bh > orig_h):
        fitting_buckets = [(w, h) for w, h in candidate_buckets if w <= orig_w and h <= orig_h]
        if fitting_buckets:
            best_bucket = max(fitting_buckets, key=lambda b: b[0] * b[1])
        else:
            min_area = min(w * h for w, h in candidate_buckets)
            floor_buckets = [(w, h) for w, h in candidate_buckets if w * h <= min_area * 1.1]
            best_bucket = min(floor_buckets, key=lambda b: bucket_score(b[0], b[1]))

    return best_bucket

def get_multi_bucket_resolutions(orig_w, orig_h, target_area=None, should_upscale=False, max_extra=0):
    primary = get_optimal_bucket(orig_w, orig_h, target_area, 64, should_upscale)
    if max_extra <= 0:
        return [primary]

    orig_ar = orig_w / max(orig_h, 1)
    max_bucket_resolution = resolve_max_bucket_resolution(target_area)
    target_area = max_bucket_resolution * max_bucket_resolution

    candidates = []
    for bucket in get_bucket_ladder(max_bucket_resolution):
        if bucket == primary:
            continue
        bw, bh = bucket
        if not should_upscale and (bw > orig_w or bh > orig_h):
            continue
        bucket_ar = bw / max(bh, 1)
        bucket_area = bw * bh
        ar_error = abs(bucket_ar - orig_ar) / max(orig_ar, 0.01)
        area_error = abs(math.log(bucket_area / target_area)) if bucket_area > 0 else 100.0
        candidates.append((ar_error * 10.0 + area_error, bucket))

    candidates.sort(key=lambda item: item[0])
    return [primary] + [bucket for _, bucket in candidates[:max_extra]]

def make_bucket_variant_metadata(base_meta, target_w, target_h, variant_index=0):
    orig_w, orig_h = base_meta["original_size"]
    scale = max(target_w / max(orig_w, 1), target_h / max(orig_h, 1))
    scaled_w = int(round(orig_w * scale))
    scaled_h = int(round(orig_h * scale))
    crop_left = max(0, (scaled_w - target_w) // 2)
    crop_top = max(0, (scaled_h - target_h) // 2)
    meta = dict(base_meta)
    meta.update({
        "target_resolution": (target_w, target_h),
        "scaled_size": (scaled_w, scaled_h),
        "crop_coords": (crop_top, crop_left),
        "bucket_variant_index": variant_index,
        "cache_suffix": "" if variant_index == 0 else f"_mb{variant_index}",
    })
    return meta

def smart_resize(image, target_w, target_h):
    orig_w, orig_h = image.size
    scale_x = target_w / max(orig_w, 1)
    scale_y = target_h / max(orig_h, 1)
    scale = max(scale_x, scale_y)
    
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))
    new_w = max(new_w, target_w)
    new_h = max(new_h, target_h)
    
    resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    crop_left = (new_w - target_w) // 2
    crop_top = (new_h - target_h) // 2
    crop_right = crop_left + target_w
    crop_bottom = crop_top + target_h
    
    cropped = resized.crop((crop_left, crop_top, crop_right, crop_bottom))
    assert cropped.size == (target_w, target_h), \
        f"smart_resize failed: expected ({target_w},{target_h}), got {cropped.size}"
    return cropped

def validate_and_assign_resolution(args):
    if len(args) >= 5:
        ip, target_area, stride, should_upscale, caption_mode = args[:5]
    else:
        ip, target_area, stride, should_upscale = args
        caption_mode = "txt"
    try:
        with Image.open(ip) as img: 
            img.verify()
        with Image.open(ip) as img:
            img.load()
            w, h = img.size
            if w <= 0 or h <= 0: 
                return None

        target_w, target_h = get_optimal_bucket(w, h, target_area, stride, should_upscale)

        scale = max(target_w / w, target_h / h)
        scaled_w = int(round(w * scale))
        scaled_h = int(round(h * scale))
        
        crop_left = max(0, (scaled_w - target_w) // 2)
        crop_top = max(0, (scaled_h - target_h) // 2)

        caption_variants = read_caption_variants_for_image(ip, caption_mode)
        caption_signature = caption_signature_from_variants(caption_variants)
        caption = caption_variants.get("txt") or caption_variants.get(CAPTION_JSON_PRIMARY_TYPE) or next(iter(caption_variants.values()))

        return {
            "ip": ip, 
            "caption": caption, 
            "caption_variants": caption_variants,
            "caption_signature": caption_signature,
            "target_resolution": (target_w, target_h),
            "original_size": (w, h), 
            "scaled_size": (scaled_w, scaled_h),
            "crop_coords": (crop_top, crop_left),
            "original_area": w * h, 
            "target_area": target_w * target_h,
            "was_upscaled": should_upscale and (w * h) < target_area
        }
    except Exception as e:
        print(f"\n[CORRUPT IMAGE OR READ ERROR] Skipping {ip}, Reason: {e}")
        return None

def caption_chunking_enabled(config):
    return bool(getattr(config, "CAPTION_CHUNKING_ENABLED", False))


def read_caption_for_image(ip, caption_mode="txt"):
    variants = read_caption_variants_for_image(ip, caption_mode)
    return variants.get("txt") or variants.get(CAPTION_JSON_PRIMARY_TYPE) or next(iter(variants.values()))


def caption_signature_from_variants(caption_variants):
    payload = {key: caption_variants[key] for key in sorted(caption_variants)}
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def caption_signature_for_image(ip, caption_mode="txt"):
    return caption_signature_from_variants(read_caption_variants_for_image(ip, caption_mode))


def read_caption_variants_for_image(ip, caption_mode="txt"):
    mode = caption_source_type(caption_mode)
    if mode == "json":
        cp = ip.with_suffix('.json')
        if not cp.exists():
            raise FileNotFoundError(f"JSON caption sidecar not found: {cp}")
        with open(cp, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"JSON caption must be an object: {cp}")
        variants = {}
        for key in CAPTION_JSON_TYPES:
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                variants[key] = value.strip()
        if not variants:
            raise ValueError(f"JSON caption {cp} must contain at least one non-empty caption key: {', '.join(CAPTION_JSON_TYPES)}")
        return variants

    cp = ip.with_suffix('.txt')
    caption = ip.stem.replace('_', ' ')
    if cp.exists():
        with open(cp, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read().strip()
            if content:
                caption = content
    return {"txt": caption}


def get_tokenizer_max_length(tokenizer):
    return int(getattr(tokenizer, "model_max_length", 77) or 77)


def get_caption_token_ids(tokenizer, caption):
    tokenized = tokenizer(caption, add_special_tokens=False, truncation=False)
    input_ids = tokenized.input_ids
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    return input_ids


def get_caption_chunk_count(caption, tokenizer):
    max_len = get_tokenizer_max_length(tokenizer)
    chunk_payload_len = max(1, max_len - 2)
    return max(1, math.ceil(len(get_caption_token_ids(tokenizer, caption)) / chunk_payload_len))


def get_max_caption_chunks_for_config(config, t1, t2):
    if not caption_chunking_enabled(config):
        return 1

    max_chunks = 1
    for dataset in config.INSTANCE_DATASETS:
        root = Path(dataset["path"])
        if not root.exists():
            continue
        for ip in (p for ext in ['.jpg', '.jpeg', '.png', '.webp', '.bmp'] for p in root.rglob(f"*{ext}")):
            try:
                caption_variants = read_caption_variants_for_image(ip, caption_source_type(config))
            except Exception as e:
                print(f"[CAPTION READ ERROR] Skipping caption chunk scan for {ip}, Reason: {e}")
                continue
            for caption in caption_variants.values():
                max_chunks = max(
                    max_chunks,
                    get_caption_chunk_count(caption, t1),
                    get_caption_chunk_count(caption, t2),
                )
    return max_chunks


def build_chunked_token_tensor(tokenizer, caption, total_chunks, device):
    max_len = get_tokenizer_max_length(tokenizer)
    payload_len = max(1, max_len - 2)
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos
    ids = get_caption_token_ids(tokenizer, caption)

    chunks = []
    for i in range(max(1, int(total_chunks or 1))):
        payload = ids[i * payload_len:(i + 1) * payload_len]
        chunk = [bos] + payload + [eos]
        chunk += [pad] * (max_len - len(chunk))
        chunks.append(chunk[:max_len])
    return torch.tensor(chunks, dtype=torch.long, device=device)


def encode_caption_chunks_sdxl(caption, t1, t2, te1, te2, device, total_chunks):
    tokens_1 = build_chunked_token_tensor(t1, caption, total_chunks, device)
    tokens_2 = build_chunked_token_tensor(t2, caption, total_chunks, device)
    enc_1_output = te1(tokens_1, output_hidden_states=True)
    enc_2_output = te2(tokens_2, output_hidden_states=True)
    hidden_1 = enc_1_output.hidden_states[-2].reshape(1, -1, enc_1_output.hidden_states[-2].shape[-1])
    hidden_2 = enc_2_output.hidden_states[-2].reshape(1, -1, enc_2_output.hidden_states[-2].shape[-1])
    return torch.cat([hidden_1, hidden_2], dim=-1), enc_2_output[0][:1]


def compute_text_embeddings_sdxl(captions, t1, t2, te1, te2, device, allow_chunking=False, total_chunks=None):
    prompt_embeds_list, pooled_prompt_embeds_list = [], []
    if allow_chunking:
        if total_chunks is None:
            total_chunks = max(
                1,
                *(max(get_caption_chunk_count(caption, t1), get_caption_chunk_count(caption, t2)) for caption in captions),
            )
        total_chunks = max(1, int(total_chunks))
    for caption in captions:
        with torch.no_grad():
            if allow_chunking:
                prompt_embeds, pooled_prompt_embeds = encode_caption_chunks_sdxl(caption, t1, t2, te1, te2, device, total_chunks)
            else:
                tokens_1 = t1(caption, padding="max_length", max_length=t1.model_max_length, truncation=True, return_tensors="pt").input_ids.to(device)
                tokens_2 = t2(caption, padding="max_length", max_length=t2.model_max_length, truncation=True, return_tensors="pt").input_ids.to(device)
                enc_1_output = te1(tokens_1, output_hidden_states=True)
                enc_2_output = te2(tokens_2, output_hidden_states=True)
                prompt_embeds = torch.cat([enc_1_output.hidden_states[-2], enc_2_output.hidden_states[-2]], dim=-1)
                pooled_prompt_embeds = enc_2_output[0]
            prompt_embeds_list.append(prompt_embeds)
            pooled_prompt_embeds_list.append(pooled_prompt_embeds)
    return torch.cat(prompt_embeds_list, dim=0), torch.cat(pooled_prompt_embeds_list, dim=0)

def get_text_conditioning_scale_range(config):
    if not bool(getattr(config, "TEXT_CONDITIONING_SCALE_ENABLED", False)):
        return 1.0, 1.0
    min_scale = float(getattr(config, "TEXT_CONDITIONING_SCALE_MIN", 1.0))
    max_scale = float(getattr(config, "TEXT_CONDITIONING_SCALE_MAX", 1.0))
    min_scale = min(max(min_scale, 0.0), 1.0)
    max_scale = min(max(max_scale, 0.0), 2.0)
    if min_scale > max_scale:
        min_scale, max_scale = max_scale, min_scale
    return min_scale, max_scale

def text_conditioning_scale_enabled(config):
    min_scale, max_scale = get_text_conditioning_scale_range(config)
    return min_scale < 1.0 or max_scale > 1.0

def null_conditioning_cache_needed(config):
    return bool(getattr(config, "UNCONDITIONAL_DROPOUT", False)) or text_conditioning_scale_enabled(config)

def get_caption_cache_options(config):
    vae_source = get_vae_source_for_config(config)
    vae_source_path = ""
    vae_source_size = None
    vae_source_mtime_ns = None
    if vae_source:
        try:
            vae_source_resolved = Path(vae_source).resolve()
            vae_source_path = str(vae_source_resolved)
            if vae_source_resolved.exists():
                stat = vae_source_resolved.stat()
                vae_source_size = stat.st_size
                vae_source_mtime_ns = stat.st_mtime_ns
        except OSError:
            vae_source_path = str(vae_source)

    return {
        "version": 13,
        "cache_schema_version": 1,
        "bucket_layout": BUCKET_LAYOUT_VERSION,
        "text_cache_float_dtype": text_cache_float_dtype_name(config),
        "vae_cache_float_dtype": vae_cache_float_dtype_name(config),
        "max_bucket_resolution": get_max_bucket_resolution_for_config(config),
        "should_upscale": bool(getattr(config, "SHOULD_UPSCALE", False)),
        "caption_embedding_layout": "fixed_total_chunks",
        "caption_source_type": caption_source_type(config),
        "caption_json_types": list(CAPTION_JSON_TYPES),
        "caption_chunking_enabled": caption_chunking_enabled(config),
        "multi_bucket_enabled": bool(getattr(config, "MULTI_BUCKET_ENABLED", False)),
        "multi_bucket_extra_buckets": int(getattr(config, "MULTI_BUCKET_EXTRA_BUCKETS", 0) or 0) if getattr(config, "MULTI_BUCKET_ENABLED", False) else 0,
        "vae_normalization_mode": getattr(config, "VAE_NORMALIZATION_MODE", "scalar"),
        "vae_shift_factor": getattr(config, "VAE_SHIFT_FACTOR", None),
        "vae_scaling_factor": getattr(config, "VAE_SCALING_FACTOR", None),
        "vae_latent_channels": getattr(config, "VAE_LATENT_CHANNELS", None),
        "vae_path": str(getattr(config, "VAE_PATH", "") or ""),
        "vae_source_path": vae_source_path,
        "vae_source_size": vae_source_size,
        "vae_source_mtime_ns": vae_source_mtime_ns,
    }

def check_if_caching_needed(config, include_null_cache=True):
    needs_caching = False
    cache_folder_name = ".precomputed_embeddings_cache_rf" if config.is_rectified_flow else ".precomputed_embeddings_cache_standard_sdxl"
    expected_cache_options = get_caption_cache_options(config)
    json_caption_mode = json_caption_mode_enabled(config)

    if include_null_cache and null_conditioning_cache_needed(config):
        if any(
            ds.get("path") and not (Path(ds["path"]) / cache_folder_name / "null_embeds.pt").exists()
            for ds in config.INSTANCE_DATASETS
        ):
            needs_caching = True

    for dataset in config.INSTANCE_DATASETS:
        root = Path(dataset["path"])
        if not root.exists(): continue
        image_paths = collect_image_paths(root)
        if not image_paths:
            cache_dir = root / cache_folder_name
            cached_te_files = list(cache_dir.glob("*_te.pt")) if cache_dir.exists() else []
            if cached_te_files:
                needs_caching = True
            elif cache_index_exists(cache_dir):
                try:
                    if load_cache_index(cache_dir).get("files"):
                        needs_caching = True
                except Exception:
                    needs_caching = True
            continue
        current_cache_stems = {cache_stem_for_image(root, p) for p in image_paths}

        cache_dir = root / cache_folder_name
        if not cache_dir.exists():
            needs_caching = True; continue
            
        if not cache_index_exists(cache_dir):
            needs_caching = True; continue
        try:
            index_data = load_cache_index(cache_dir)
            if not cache_image_layout_options_match(index_data.get("cache_options"), expected_cache_options):
                needs_caching = True
            indexed_files = index_data.get("files", [])
            if any("scaled_size" not in item for item in indexed_files):
                needs_caching = True
            if len(indexed_files) < len(image_paths):
                needs_caching = True
            indexed_base_stems = {
                cache_base_stem_from_te_path(te_path)
                for item in indexed_files
                for te_path in te_paths_for_index_item(item)
            }
            indexed_base_stems.discard(None)
            if not current_cache_stems.issubset(indexed_base_stems):
                needs_caching = True
            if any(stem not in current_cache_stems for stem in indexed_base_stems):
                needs_caching = True
            for item in indexed_files:
                te_paths = te_paths_for_index_item(item)
                lat_path = item.get("lat_path")
                if not te_paths or not lat_path or not Path(lat_path).exists() or any(not Path(p).exists() for p in te_paths):
                    needs_caching = True
                    break
                try:
                    text_payloads = [
                        torch.load(p, map_location="cpu", weights_only=True)
                        for p in te_paths
                    ]
                    if any(
                        not sdxl_text_cache_compatible_options(payload.get("cache_options"), expected_cache_options)
                        for payload in text_payloads
                    ):
                        needs_caching = True
                        break
                    lat_payload = torch.load(lat_path, map_location="cpu", weights_only=True)
                    if not isinstance(lat_payload, dict):
                        needs_caching = True
                        break
                    if not sdxl_lat_cache_compatible_options(lat_payload.get("cache_options"), expected_cache_options):
                        needs_caching = True
                        break
                except Exception:
                    needs_caching = True
                    break
                relative_path = item.get("relative_path")
                cached_signature = item.get("caption_signature")
                if relative_path:
                    try:
                        image_path = root / relative_path
                        stat_match = cached_file_signatures_match(item, image_path, caption_source_type(config))
                        if stat_match is False:
                            needs_caching = True
                            break
                        if stat_match is None and caption_signature_for_image(image_path, caption_source_type(config)) != cached_signature:
                            needs_caching = True
                            break
                    except Exception:
                        needs_caching = True
                        break
        except Exception:
            needs_caching = True

        cached_te_files = list(cache_dir.glob("*_te.pt"))
        cached_base_stems = {
            cache_base_stem_from_te_path(f)
            for f in cached_te_files
        }
        cached_base_stems.discard(None)
        if any(stem not in current_cache_stems for stem in cached_base_stems):
            needs_caching = True
        if not current_cache_stems.issubset(cached_base_stems):
            needs_caching = True
        expected_te_count = 0
        try:
            max_bucket_resolution = get_max_bucket_resolution_for_config(config)
            max_bucket_area = max_bucket_resolution * max_bucket_resolution
            multi_bucket_extra = (
                max(0, int(getattr(config, "MULTI_BUCKET_EXTRA_BUCKETS", 0) or 0))
                if getattr(config, "MULTI_BUCKET_ENABLED", False)
                else 0
            )
            for image_path in image_paths:
                caption_variant_count = (
                    len(read_caption_variants_for_image(image_path, caption_source_type(config)))
                    if json_caption_mode
                    else 1
                )
                with Image.open(image_path) as img:
                    bucket_count = len(get_multi_bucket_resolutions(
                        img.width,
                        img.height,
                        max_bucket_area,
                        getattr(config, "SHOULD_UPSCALE", False),
                        multi_bucket_extra,
                    ))
                expected_te_count += caption_variant_count * bucket_count
        except Exception:
            needs_caching = True
            expected_te_count = len(image_paths)
        if len(cached_te_files) < expected_te_count:
            needs_caching = True
        else:
            for f in cached_te_files[: min(len(cached_te_files), 10)]:
                try:
                    data_te = torch.load(f, map_location="cpu", weights_only=True)
                    if not sdxl_text_cache_compatible_options(data_te.get("cache_options"), expected_cache_options):
                        needs_caching = True
                        break
                except Exception:
                    needs_caching = True
                    break
    return needs_caching

def load_unet_robust(path, compute_dtype):
    print(f"INFO: Loading UNet from: {Path(path).name}")
    in_channels = 4
    out_channels = 4
    try:
        from safetensors import safe_open
        with safe_open(path, framework="pt", device="cpu") as f:
            key_in = "model.diffusion_model.input_blocks.0.0.weight"
            key_out = "model.diffusion_model.out.2.weight"
            if key_in in f.keys():
                shape_in = f.get_slice(key_in).get_shape()
                in_channels = shape_in[1]
            if key_out in f.keys():
                shape_out = f.get_slice(key_out).get_shape()
                out_channels = shape_out[0]
    except Exception as e:
        print(f"WARNING: Could not peek into safetensors for channel sizes, falling back to defaults. Error: {e}")

    print(f"INFO: Detected UNet configuration - in_channels: {in_channels}, out_channels: {out_channels}")

    try:
        unet = UNet2DConditionModel.from_single_file(
            path,
            torch_dtype=compute_dtype,
            low_cpu_mem_usage=True,
            in_channels=in_channels,
            out_channels=out_channels
        )
        print(f"INFO: Loaded UNet (Channels: In={unet.config.in_channels} / Out={unet.config.out_channels})")
        return unet
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load UNet. {e}")
        raise e

def load_vae_robust(path, device, target_channels=None):
    print(f"INFO: Attempting to load VAE from: {path}")
    if target_channels is None:
        try:
            tensors = load_file(path, device="cpu")
            for k in ["first_stage_model.quant_conv.weight", "quant_conv.weight"]:
                if k in tensors:
                    target_channels = tensors[k].shape[0] // 2
                    break
        except Exception: pass

    target_channels = target_channels or 4
    try:
        if target_channels != 4:
            vae = AutoencoderKL.from_single_file(path, torch_dtype=torch.float32, latent_channels=target_channels, ignore_mismatched_sizes=True, low_cpu_mem_usage=False)
        else:
            vae = AutoencoderKL.from_single_file(path, torch_dtype=torch.float32, low_cpu_mem_usage=True)
        print(f"INFO: Successfully loaded VAE with {target_channels} channels.")
        return vae.to(device)
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load VAE: {e}")
        raise e

def find_tensor_by_suffix(safetensors_path, suffixes):
    with safe_open(str(safetensors_path), framework="pt", device="cpu") as f:
        keys = list(f.keys())
        for suffix in suffixes:
            matches = [k for k in keys if k == suffix or k.endswith("." + suffix)]
            if matches:
                key = sorted(matches, key=len)[0]
                return f.get_tensor(key).float(), key
    return None, None

def extract_flux_bn_stats_from_safetensor(safetensors_path):
    mean, mean_key = find_tensor_by_suffix(safetensors_path, BN_MEAN_SUFFIXES)
    var, var_key = find_tensor_by_suffix(safetensors_path, BN_VAR_SUFFIXES)

    if mean is None or var is None:
        raise RuntimeError(
            f"Could not find Flux BN stats in {safetensors_path}. "
            "Expected keys ending with bn.running_mean and bn.running_var."
        )
    if mean.numel() != 128 or var.numel() != 128:
        raise RuntimeError(
            f"Flux BN stats found but wrong shape. "
            f"mean={tuple(mean.shape)}, var={tuple(var.shape)}. Expected 128 elements."
        )

    print(
        f"INFO: Loaded Flux VAE BN stats from {Path(safetensors_path).name}\n"
        f"      mean key: {mean_key}\n"
        f"      var key:  {var_key}\n"
        f"      mean range: [{mean.min().item():+.5f}, {mean.max().item():+.5f}]\n"
        f"      var  range: [{var.min().item():+.5f}, {var.max().item():+.5f}]"
    )
    return mean, var

def flux_bn32_to_bn128_layout(latents):
    if latents.dim() != 4 or latents.shape[1] != 32:
        raise RuntimeError(f"flux_bn32 expects [N, 32, H, W] latents before BN, got shape {tuple(latents.shape)}")
    if latents.shape[2] % 2 != 0 or latents.shape[3] % 2 != 0:
        raise RuntimeError(f"flux_bn32 requires even latent height/width, got shape {tuple(latents.shape)}")

    n, c, h, w = latents.shape
    return (
        latents.view(n, c, h // 2, 2, w // 2, 2)
        .permute(0, 1, 3, 5, 2, 4)
        .reshape(n, c * 4, h // 2, w // 2)
    )

def flux_bn128_to_bn32_layout(latents):
    if latents.dim() != 4 or latents.shape[1] != 128:
        raise RuntimeError(f"flux_bn32 decode expects [N, 128, H, W] BN latents, got shape {tuple(latents.shape)}")

    n, c, h, w = latents.shape
    return (
        latents.view(n, c // 4, 2, 2, h, w)
        .permute(0, 1, 4, 2, 5, 3)
        .reshape(n, c // 4, h * 2, w * 2)
    )

def apply_flux_bn32_norm(latents, mean_128, var_128):
    latents_bn128 = flux_bn32_to_bn128_layout(latents)
    mean_128 = mean_128.to(device=latents_bn128.device, dtype=latents_bn128.dtype)
    var_128 = var_128.to(device=latents_bn128.device, dtype=latents_bn128.dtype)
    latents_bn128 = F.batch_norm(latents_bn128, mean_128, var_128, training=False, momentum=0.1, eps=FLUX_BN_EPS)
    return flux_bn128_to_bn32_layout(latents_bn128)

def invert_flux_bn32_norm(latents, mean_128, var_128):
    latents_bn128 = flux_bn32_to_bn128_layout(latents)
    mean_128 = mean_128.to(device=latents.device, dtype=latents.dtype).view(1, -1, 1, 1)
    sigma_128 = torch.sqrt(var_128.to(device=latents.device, dtype=latents.dtype).view(1, -1, 1, 1) + FLUX_BN_EPS)
    return flux_bn128_to_bn32_layout(latents_bn128 * sigma_128 + mean_128)

def get_vae_source_for_config(config):
    vae_path = getattr(config, "VAE_PATH", None)
    if vae_path and Path(vae_path).exists():
        return vae_path
    return getattr(config, "SINGLE_FILE_CHECKPOINT_PATH", None)

def denormalize_latents_for_vae_decode(latents, config, vae=None, bn_mean_128=None, bn_var_128=None):
    mode = str(getattr(config, "VAE_NORMALIZATION_MODE", "scalar")).lower()

    if mode == "flux_bn32":
        if bn_mean_128 is None or bn_var_128 is None:
            vae_source = get_vae_source_for_config(config)
            if not vae_source or not Path(vae_source).exists():
                raise RuntimeError("VAE_NORMALIZATION_MODE='flux_bn32' requires VAE_PATH or SINGLE_FILE_CHECKPOINT_PATH.")
            bn_mean_128, bn_var_128 = extract_flux_bn_stats_from_safetensor(vae_source)
        return invert_flux_bn32_norm(latents, bn_mean_128, bn_var_128)

    if mode != "scalar":
        raise RuntimeError(f"Unknown VAE_NORMALIZATION_MODE: {mode}")

    shift = getattr(config, "VAE_SHIFT_FACTOR", None)
    scale = getattr(config, "VAE_SCALING_FACTOR", None)
    if scale is None and vae is not None:
        scale = getattr(vae.config, "scaling_factor", 1.0)
    if shift is None and vae is not None:
        shift = getattr(vae.config, "shift_factor", None)

    scale = 1.0 if scale is None else scale
    if shift is not None:
        return latents / scale + shift
    return latents / scale

def precompute_and_cache_latents(config, t1, t2, te1, te2, vae, device):
    if not check_if_caching_needed(config):
        print("\n" + "="*60 + "\nINFO: Datasets already cached.\n" + "="*60 + "\n")
        return

    cache_folder_name = ".precomputed_embeddings_cache_rf" if config.is_rectified_flow else ".precomputed_embeddings_cache_standard_sdxl"
    expected_cache_options = get_caption_cache_options(config)
    text_cache_dtype = text_cache_float_dtype(config)
    vae_cache_dtype = vae_cache_float_dtype(config)
    caption_mode = caption_source_type(config)
    json_caption_mode = json_caption_mode_enabled(caption_mode)
    print(
        "INFO: SDXL cache precision: "
        f"text={text_cache_float_dtype_name(config)}, "
        f"vae={vae_cache_float_dtype_name(config)}."
    )

    vae.to(device, dtype=torch.float32)
    vae.enable_tiling()
    vae.enable_slicing()
    print(f"INFO: VAE shift={vae.config.shift_factor}, scale={vae.config.scaling_factor}, channels={vae.config.latent_channels}")

    vae_norm_mode = str(getattr(config, "VAE_NORMALIZATION_MODE", "scalar")).lower()
    bn_mean_128 = None
    bn_var_128 = None
    if vae_norm_mode == "flux_bn32":
        vae_source = get_vae_source_for_config(config)
        if not vae_source or not Path(vae_source).exists():
            raise RuntimeError("VAE_NORMALIZATION_MODE='flux_bn32' requires VAE_PATH or SINGLE_FILE_CHECKPOINT_PATH.")
        bn_mean_128, bn_var_128 = extract_flux_bn_stats_from_safetensor(vae_source)
        print(
            "INFO: Using ComfyUI-style Flux BN32 latent normalization\n"
            "      applies BN in [N,128,H/2,W/2] layout, then stores [N,32,H,W]\n"
            f"      mean_128 range: [{bn_mean_128.min().item():+.5f}, {bn_mean_128.max().item():+.5f}]\n"
            f"      var_128 range:  [{bn_var_128.min().item():+.5f}, {bn_var_128.max().item():+.5f}]"
        )
    elif vae_norm_mode != "scalar":
        raise RuntimeError(f"Unknown VAE_NORMALIZATION_MODE: {vae_norm_mode}")

    te1.to(device)
    te2.to(device)
    allow_caption_chunking = caption_chunking_enabled(config)
    caption_total_chunks = get_max_caption_chunks_for_config(config, t1, t2) if allow_caption_chunking else 1
    if allow_caption_chunking:
        print(
            f"INFO: Caption chunking enabled for cached text embeddings: "
            f"{caption_total_chunks} chunk(s), {caption_total_chunks * min(get_tokenizer_max_length(t1), get_tokenizer_max_length(t2))} tokens max."
        )

    if null_conditioning_cache_needed(config):
        with torch.no_grad():
            null_embeds, null_pooled = compute_text_embeddings_sdxl(
                [""],
                t1, t2, te1, te2, device,
                allow_chunking=allow_caption_chunking,
                total_chunks=caption_total_chunks,
            )
    else:
        null_embeds, null_pooled = None, None

    if (
        null_embeds is not None
        and null_pooled is not None
        and not check_if_caching_needed(config, include_null_cache=False)
    ):
        created = 0
        for dataset in config.INSTANCE_DATASETS:
            root = Path(dataset["path"])
            cache_dir = root / cache_folder_name
            if cache_index_exists(cache_dir) and not (cache_dir / "null_embeds.pt").exists():
                torch.save(
                    {
                        "embeds": null_embeds.to(dtype=text_cache_dtype).cpu(),
                        "pooled": null_pooled.to(dtype=text_cache_dtype).cpu(),
                    },
                    cache_dir / "null_embeds.pt",
                )
                created += 1
        print(f"INFO: Created SDXL null conditioning cache for {created} dataset(s).")
        return

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    for dataset in config.INSTANCE_DATASETS:
        root = Path(dataset["path"])
        cache_dir = root / cache_folder_name
        cache_dir.mkdir(exist_ok=True)

        paths = collect_image_paths(root)
        current_cache_stems = {cache_stem_for_image(root, p) for p in paths}
        multi_bucket_extra = (
            max(0, int(getattr(config, "MULTI_BUCKET_EXTRA_BUCKETS", 0) or 0))
            if getattr(config, "MULTI_BUCKET_ENABLED", False)
            else 0
        )
        if multi_bucket_extra > 0:
            print(f"INFO: Multi-bucket cache enabled: up to {multi_bucket_extra} extra bucket(s) per image.")
        force_recaching = bool(getattr(config, "REBUILD_CACHE", False))
        if cache_index_exists(cache_dir) and not force_recaching:
            try:
                existing_index = load_cache_index(cache_dir)
                if not cache_image_layout_options_match(existing_index.get("cache_options"), expected_cache_options):
                    print(
                        f"INFO: SDXL cache options changed for {root.name}; "
                        "reusing compatible cached files and filling only missing variants."
                    )
            except Exception:
                print(f"INFO: SDXL cache index for {root.name} is unreadable; rebuilding index from cache files.")

        if force_recaching:
            stale_files = list(cache_dir.glob("*_te*.pt")) + list(cache_dir.glob("*_lat.pt"))
            for stale_file in stale_files:
                if not stale_file.exists():
                    continue
                try:
                    stale_file.unlink()
                except OSError as e:
                    print(f"WARNING: Could not remove stale cache file {stale_file}: {e}")
        else:
            stale_files = [
                f for f in cache_dir.glob("*.pt")
                if f.name not in {"null_embeds.pt", "dataset_index.pt"} and cache_base_stem_from_cache_path(f) not in current_cache_stems
            ]
            if stale_files:
                print(f"INFO: Removing {len(stale_files)} stale SDXL cache item(s) for deleted images in {root.name}.")
            for stale_file in stale_files:
                try:
                    stale_file.unlink()
                except OSError as e:
                    print(f"WARNING: Could not remove stale cache file {stale_file}: {e}")

        if null_embeds is not None and null_pooled is not None:
            torch.save(
                {
                    "embeds": null_embeds.to(dtype=text_cache_dtype).cpu(),
                    "pooled": null_pooled.to(dtype=text_cache_dtype).cpu(),
                },
                cache_dir / "null_embeds.pt",
            )

        if paths:
            with Pool(processes=min(cpu_count(), 8)) as pool:
                max_bucket_resolution = get_max_bucket_resolution_for_config(config)
                max_bucket_area = max_bucket_resolution * max_bucket_resolution
                print(f"INFO: Validating SDXL images and assigning buckets for {root.name} ({len(paths)} image(s))...")
                results = list(tqdm(pool.imap(validate_and_assign_resolution, [(p, max_bucket_area, 64, config.SHOULD_UPSCALE, caption_mode) for p in paths]), total=len(paths)))

            expanded_results = []
            for m in (r for r in results if r):
                buckets_for_image = get_multi_bucket_resolutions(
                    m["original_size"][0],
                    m["original_size"][1],
                    max_bucket_area,
                    config.SHOULD_UPSCALE,
                    multi_bucket_extra,
                )
                for variant_index, (target_w, target_h) in enumerate(buckets_for_image):
                    expanded_results.append(make_bucket_variant_metadata(m, target_w, target_h, variant_index))

            cache_caption_types = caption_types_for_cache(json_caption_mode)
            text_jobs = []
            lat_jobs = []
            reused_text_count = 0
            reused_lat_count = 0
            expected_text_count = 0
            expected_cache_files = set()
            for m in expanded_results:
                caption_variants = m.get("caption_variants") or {"txt": m["caption"]}
                item_caption_types = tuple(key for key in cache_caption_types if key in caption_variants)
                text_paths, lat_path = expected_cache_paths_for_metadata(root, cache_dir, m, item_caption_types, json_caption_mode)
                expected_cache_files.add(lat_path.resolve())
                expected_text_count += len(item_caption_types)
                for caption_type in item_caption_types:
                    te_path = text_paths[caption_type]
                    expected_cache_files.add(te_path.resolve())
                    caption = caption_variants[caption_type]
                    if te_path.exists() and sdxl_text_cache_valid(te_path, root, m, caption_type, caption, text_cache_dtype, expected_cache_options):
                        reused_text_count += 1
                    else:
                        text_jobs.append((m, caption_type, caption, te_path))

                if lat_path.exists() and sdxl_lat_cache_valid(lat_path, root, m, vae_cache_dtype, expected_cache_options):
                    reused_lat_count += 1
                else:
                    lat_jobs.append((m, lat_path))

            obsolete_files = [
                f for f in cache_dir.glob("*.pt")
                if f.name not in {"null_embeds.pt", "dataset_index.pt"}
                and cache_base_stem_from_cache_path(f) in current_cache_stems
                and f.resolve() not in expected_cache_files
            ]
            if obsolete_files:
                print(f"INFO: Removing {len(obsolete_files)} obsolete SDXL cache variant(s) for {root.name}.")
            for f in obsolete_files:
                try:
                    f.unlink()
                except OSError as e:
                    print(f"WARNING: Could not remove stale cache file {f}: {e}")

            if text_jobs or lat_jobs:
                print(
                    f"INFO: SDXL cache reuse for {root.name}: "
                    f"{reused_text_count}/{expected_text_count} text item(s), "
                    f"{reused_lat_count}/{len(expanded_results)} latent item(s)."
                )

            if text_jobs:
                text_batch_size = max(1, int(getattr(config, "CACHING_BATCH_SIZE", 1) or 1) * len(cache_caption_types))
                with tqdm(total=len(text_jobs), desc=f"Caching SDXL text {root.name}", unit="item") as text_pbar:
                    for i in range(0, len(text_jobs), text_batch_size):
                        batch_jobs = text_jobs[i:i + text_batch_size]
                        embeds_all, pooled_all = compute_text_embeddings_sdxl(
                            [caption for _, _, caption, _ in batch_jobs],
                            t1, t2, te1, te2, device,
                            allow_chunking=allow_caption_chunking,
                            total_chunks=caption_total_chunks,
                        )
                        for embed_index, (m, caption_type, caption, te_path) in enumerate(batch_jobs):
                            w, h = m["target_resolution"]
                            torch.save({
                                "original_stem": m["ip"].stem,
                                "relative_path": str(m["ip"].relative_to(root)),
                                "image_file_signature": image_file_signature(m["ip"]),
                                "caption_file_signature": caption_file_signature_for_image(m["ip"], caption_mode),
                                "original_size": m["original_size"],
                                "scaled_size": m.get("scaled_size", m["original_size"]),
                                "target_size": (w, h),
                                "crop_coords": m.get("crop_coords", (0, 0)),
                                "bucket_variant_index": m.get("bucket_variant_index", 0),
                                "caption_type": caption_type,
                                "caption": caption,
                                "caption_signature": m.get("caption_signature"),
                                "embeds": embeds_all[embed_index].to(dtype=text_cache_dtype).cpu(),
                                "pooled": pooled_all[embed_index].to(dtype=text_cache_dtype).cpu(),
                                "cache_options": expected_cache_options,
                                "vae_normalization_mode": vae_norm_mode,
                                "vae_shift": vae.config.shift_factor,
                                "vae_scale": vae.config.scaling_factor,
                                "flux_bn_eps": FLUX_BN_EPS if vae_norm_mode == "flux_bn32" else None,
                            }, te_path)
                            text_pbar.update(1)

            if lat_jobs:
                grouped = defaultdict(list)
                for m, lat_path in lat_jobs:
                    grouped[m["target_resolution"]].append((m, lat_path))

                batches = [
                    (res, grouped[res][i:i + config.CACHING_BATCH_SIZE])
                    for res in grouped
                    for i in range(0, len(grouped[res]), config.CACHING_BATCH_SIZE)
                ]
                random.shuffle(batches)

                with tqdm(total=len(lat_jobs), desc=f"Caching SDXL VAE {root.name}", unit="item") as vae_pbar:
                    for batch_idx, ((w, h), batch_meta) in enumerate(batches):
                        images_global, valid_meta_final, skipped_meta = [], [], []
                        for m, lat_path in batch_meta:
                            try:
                                with Image.open(m['ip']) as img:
                                    img_resized = smart_resize(fix_alpha_channel(img), w, h)
                                    images_global.append(transform(img_resized))
                                    valid_meta_final.append((m, lat_path))
                            except Exception as e:
                                skipped_meta.append((m, lat_path))
                                tqdm.write(f"[SKIP] {m['ip'].name}: {e}")

                        for _, lat_path in skipped_meta:
                            try:
                                Path(lat_path).unlink()
                            except OSError:
                                pass
                            vae_pbar.update(1)

                        if not images_global:
                            continue

                        with torch.no_grad():
                            latents_global = vae.encode(torch.stack(images_global).to(device, dtype=torch.float32)).latent_dist.mean
                            raw_min = latents_global.min().item()
                            raw_max = latents_global.max().item()
                            raw_mean = latents_global.mean().item()
                            raw_std = latents_global.std().item()

                            if vae_norm_mode == "flux_bn32":
                                if latents_global.shape[1] != 32:
                                    raise RuntimeError(
                                        f"flux_bn32 expects 32-channel latents, got shape {tuple(latents_global.shape)}"
                                    )
                                latents_global = apply_flux_bn32_norm(latents_global, bn_mean_128, bn_var_128)
                            elif getattr(vae.config, 'shift_factor', None) is not None:
                                latents_global = (latents_global - vae.config.shift_factor) * vae.config.scaling_factor
                            else:
                                latents_global = latents_global * vae.config.scaling_factor

                            norm_min = latents_global.min().item()
                            norm_max = latents_global.max().item()
                            norm_mean = latents_global.mean().item()
                            norm_std = latents_global.std().item()

                        latents_global = latents_global.cpu()
                        for j, (m, lat_path) in enumerate(valid_meta_final):
                            w, h = m["target_resolution"]
                            torch.save({
                                "latents": latents_global[j].to(dtype=vae_cache_dtype).cpu(),
                                "relative_path": str(m["ip"].relative_to(root)),
                                "image_file_signature": image_file_signature(m["ip"]),
                                "caption_file_signature": caption_file_signature_for_image(m["ip"], caption_mode),
                                "original_size": m["original_size"],
                                "scaled_size": m.get("scaled_size", m["original_size"]),
                                "target_size": (w, h),
                                "crop_coords": m.get("crop_coords", (0, 0)),
                                "bucket_variant_index": m.get("bucket_variant_index", 0),
                                "caption_signature": m.get("caption_signature"),
                                "cache_options": expected_cache_options,
                                "vae_normalization_mode": vae_norm_mode,
                                "vae_shift": vae.config.shift_factor,
                                "vae_scale": vae.config.scaling_factor,
                                "flux_bn_eps": FLUX_BN_EPS if vae_norm_mode == "flux_bn32" else None,
                            }, lat_path)
                            vae_pbar.update(1)

                        if batch_idx % 20 == 0:
                            torch.cuda.empty_cache()

        print(f"INFO: Building fast JSON index for {root.name}...")
        index_data = []
        cached_files = list(cache_dir.glob("*_te.pt"))
        if json_caption_mode:
            grouped_te_files = defaultdict(dict)
            for f in cached_files:
                item_stem = cache_item_stem_from_te_path(f)
                if item_stem is None:
                    continue
                try:
                    caption_type = torch.load(f, map_location='cpu', weights_only=True).get("caption_type")
                    if caption_type in CAPTION_JSON_TYPES:
                        grouped_te_files[item_stem][caption_type] = f
                except Exception as e:
                    print(f"WARNING: Could not inspect cached text file {f}: {e}")

            iterable_index_items = []
            for item_stem, variant_paths in grouped_te_files.items():
                if not variant_paths:
                    continue
                primary_variant = variant_paths.get(CAPTION_JSON_PRIMARY_TYPE) or next(iter(variant_paths.values()))
                iterable_index_items.append((item_stem, primary_variant, variant_paths))
        else:
            iterable_index_items = [
                (cache_item_stem_from_te_path(f), f, None)
                for f in cached_files
                if cache_item_stem_from_te_path(f) is not None
            ]

        for item_stem, f, variant_paths in iterable_index_items:
            try:
                if cache_base_stem_from_te_path(f) not in current_cache_stems:
                    remove_cache_pair_for_te_path(f)
                    continue
                lat_path = cache_dir / f"{item_stem}_lat.pt"
                if not lat_path.exists():
                    print(f"WARNING: Skipping cached text file with missing latent: {f}")
                    continue
                data_te = torch.load(f, map_location='cpu', weights_only=True)
                relative_path = data_te.get("relative_path")
                if relative_path and not (root / relative_path).exists():
                    remove_cache_pair_for_te_path(f)
                    continue

                index_item = {
                    "te_path": str(f),
                    "lat_path": str(lat_path),
                    "relative_path": relative_path,
                    "image_file_signature": data_te.get("image_file_signature"),
                    "caption_file_signature": data_te.get("caption_file_signature"),
                    "target_size": data_te.get('target_size'),
                    "original_size": data_te.get('original_size'),
                    "scaled_size": data_te.get('scaled_size', data_te.get('original_size')),
                    "crop_coords": data_te.get('crop_coords', (0,0)),
                    "bucket_variant_index": data_te.get("bucket_variant_index", 0),
                    "caption_signature": data_te.get("caption_signature"),
                }
                if variant_paths:
                    index_item["caption_variants"] = caption_variant_index(variant_paths)
                index_data.append(index_item)
            except Exception as e:
                print(f"WARNING: Could not index {f}: {e}")
                
        index_path = save_cache_index(cache_dir, {"version": 13, "cache_options": expected_cache_options, "files": index_data})
        print(f"INFO: Saved dataset index to {index_path}")

    te1.cpu(); te2.cpu(); gc.collect(); torch.cuda.empty_cache()


class ImageTextLatentDataset(Dataset):
    SAMPLE_INDEX_BITS = 32
    SAMPLE_INDEX_MASK = (1 << SAMPLE_INDEX_BITS) - 1

    def __init__(self, config):
        self.items = []
        self.bucket_keys = []
        self.seed = config.SEED if config.SEED else 42
        self.json_caption_mode = json_caption_mode_enabled(config)
        self.caption_weights = get_json_caption_weights(config)
        cache_folder_name = ".precomputed_embeddings_cache_rf" if config.is_rectified_flow else ".precomputed_embeddings_cache_standard_sdxl"
        for ds in config.INSTANCE_DATASETS:
            root = Path(ds["path"])
            cache_dir = root / cache_folder_name
            if cache_index_exists(cache_dir):
                index_data = load_cache_index(cache_dir)
                    
                repeats = int(ds.get("repeats", 1))
                stable_items = sorted(index_data["files"], key=stable_cache_item_key)
                for _ in range(repeats):
                    for item in stable_items:
                        self.items.append(item)
                        self.bucket_keys.append(tuple(item["target_size"]))
            else:
                print(f"WARNING: Index missing at {cache_dir}. Please re-run caching!")

        if not self.items: raise ValueError("No cached files found.")

        combined = list(zip(self.items, self.bucket_keys))
        random.Random(self.seed).shuffle(combined)
        self.items, self.bucket_keys = zip(*combined)
        self.items = list(self.items)
        self.bucket_keys = list(self.bucket_keys)
        self.null_embeds = None
        self.null_pooled = None
        self.cond_scale_min, self.cond_scale_max = get_text_conditioning_scale_range(config)
        self.cond_scale_enabled = self.cond_scale_min < 1.0 or self.cond_scale_max > 1.0
        self.dropout_prob = (
            min(max(float(getattr(config, "UNCONDITIONAL_DROPOUT_CHANCE", 0.0)), 0.0), 1.0)
            if getattr(config, "UNCONDITIONAL_DROPOUT", False)
            else 0.0
        )
        if self.dropout_prob > 0 or self.cond_scale_enabled:
            try:
                null_data = torch.load(Path(config.INSTANCE_DATASETS[0]["path"]) / cache_folder_name / "null_embeds.pt", map_location="cpu", weights_only=True)
                self.null_embeds = null_data["embeds"].squeeze(0) if null_data["embeds"].dim() == 3 else null_data["embeds"]
                self.null_pooled = null_data["pooled"].squeeze(0) if null_data["pooled"].dim() == 2 else null_data["pooled"]
            except Exception:
                self.dropout_prob = 0.0
                self.cond_scale_enabled = False

    def __len__(self): return len(self.items)

    @classmethod
    def pack_sample_index(cls, dataset_index, sample_index):
        dataset_index = int(dataset_index)
        sample_index = int(sample_index)
        if dataset_index < 0 or dataset_index > cls.SAMPLE_INDEX_MASK:
            raise ValueError(f"Dataset index is too large to pack deterministically: {dataset_index}")
        return (sample_index << cls.SAMPLE_INDEX_BITS) | dataset_index

    @classmethod
    def unpack_sample_index(cls, packed_index):
        packed_index = int(packed_index)
        dataset_index = packed_index & cls.SAMPLE_INDEX_MASK
        sample_index = packed_index >> cls.SAMPLE_INDEX_BITS
        return dataset_index, sample_index

    def _rng_for_sample(self, dataset_index, sample_index):
        payload = f"{self.seed}:sdxl-sample:{int(sample_index)}:{int(dataset_index)}".encode("utf-8")
        digest = hashlib.sha256(payload).digest()
        return random.Random(int.from_bytes(digest[:8], "little"))

    def _load_tensor(self, path):
        if not path: return None
        return torch.load(path, map_location="cpu", weights_only=True)

    def _load_latent_payload(self, path):
        payload = self._load_tensor(path)
        if isinstance(payload, dict):
            return payload.get("latents")
        return payload

    def _resize_null_embeds(self, target_len, dtype):
        null_embeds = self.null_embeds
        if null_embeds is None:
            return null_embeds
        if null_embeds.shape[0] == target_len:
            return null_embeds.to(dtype=dtype)

        null_len = null_embeds.shape[0]
        if target_len < null_len:
            return null_embeds[:target_len].to(dtype=dtype)

        chunk_len = CLIP_CHUNK_TOKEN_COUNT if null_len >= CLIP_CHUNK_TOKEN_COUNT else null_len
        if chunk_len <= 0 or null_len % chunk_len != 0:
            pad = null_embeds[-1:].expand(target_len - null_len, -1)
            return torch.cat([null_embeds, pad], dim=0).to(dtype=dtype)

        null_chunk = null_embeds[-chunk_len:]
        missing = target_len - null_len
        full_chunks, partial_chunk = divmod(missing, chunk_len)
        parts = [null_embeds]
        if full_chunks:
            parts.append(null_chunk.repeat(full_chunks, 1))
        if partial_chunk:
            parts.append(null_chunk[:partial_chunk])
        return torch.cat(parts, dim=0).to(dtype=dtype)

    def _align_null_embeds(self, embeds):
        null_embeds = self.null_embeds
        if null_embeds is None or embeds.shape == null_embeds.shape:
            return embeds, null_embeds
        if embeds.dim() != 2 or null_embeds.dim() != 2 or embeds.shape[1] != null_embeds.shape[1]:
            return embeds, null_embeds

        embed_len = embeds.shape[0]
        null_len = null_embeds.shape[0]
        if embed_len < null_len:
            pad = self._resize_null_embeds(null_len, embeds.dtype)[embed_len:null_len]
            embeds = torch.cat([embeds, pad], dim=0)
        elif embed_len > null_len:
            null_embeds = self._resize_null_embeds(embed_len, null_embeds.dtype)
        return embeds, null_embeds

    def __getitem__(self, i):
        try:
            dataset_index, sample_index = self.unpack_sample_index(i)
            rng = self._rng_for_sample(dataset_index, sample_index)
            item_data = self.items[dataset_index]
            path_te = selected_caption_variant_path(
                item_data,
                rng,
                self.caption_weights,
                enabled=self.json_caption_mode,
            )
            path_lat = item_data["lat_path"]

            data_te = self._load_tensor(path_te)
            latents = self._load_latent_payload(path_lat)
            embeds, pooled = data_te["embeds"], data_te["pooled"]

            if torch.isnan(latents).any() or torch.isinf(latents).any(): return None

            item = {
                "latents": latents,
                "embeds": embeds.squeeze(0) if embeds.dim() == 3 else embeds,
                "pooled": pooled.squeeze(0) if pooled.dim() == 2 else pooled,
                "original_sizes": item_data["original_size"], 
                "scaled_sizes": item_data.get("scaled_size", item_data["original_size"]),
                "target_sizes": item_data["target_size"],
                "crop_coords": item_data.get("crop_coords", (0, 0)), 
                "latent_path": path_te,
                "image_key": item_data.get("relative_path", path_lat),
            }

            if self.dropout_prob > 0 and rng.random() < self.dropout_prob:
                _, null_embeds = self._align_null_embeds(item["embeds"])
                item["embeds"], item["pooled"] = null_embeds, self.null_pooled
            elif self.cond_scale_enabled:
                scale = rng.uniform(self.cond_scale_min, self.cond_scale_max)
                embeds, null_embeds = self._align_null_embeds(item["embeds"])
                item["embeds"] = null_embeds + (embeds - null_embeds) * scale
                item["pooled"] = self.null_pooled + (item["pooled"] - self.null_pooled) * scale

            return item
        except Exception as e:
            print(f"[DATASET] Failed to load item {i}: {e}")
            return None


class TimestepSampler:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.total_tickets_needed = config.MAX_TRAIN_STEPS * config.BATCH_SIZE
        self.seed = config.SEED if config.SEED else 42
        self.is_rectified_flow = config.is_rectified_flow
        self.bin_ranges = []
        allocation = getattr(config, 'TIMESTEP_ALLOCATION', None)
        self.ticket_pool = self._build_ticket_pool(allocation)
        self.pool_index = 0

    def _build_ticket_pool(self, allocation):
        pool, self.bin_ranges = build_timestep_ticket_pool(
            allocation,
            self.total_tickets_needed,
            1000,
            self.seed,
            bool(getattr(self.config, "TIMESTEP_STRATIFIED_SAMPLING", False)),
        )
        return pool

    def set_current_step(self, micro_step):
        self.pool_index = (micro_step * self.config.BATCH_SIZE) % len(self.ticket_pool)

    def _sample_from_pool(self):
        if self.pool_index >= len(self.ticket_pool): self.pool_index = 0
        index = self.ticket_pool[self.pool_index]
        self.pool_index += 1
        return index

    def state_dict(self):
        return {
            "pool_index": self.pool_index,
        }

    def load_state_dict(self, state):
        if not isinstance(state, dict):
            return
        self.pool_index = int(state.get("pool_index", self.pool_index)) % len(self.ticket_pool)

    def sample(self, batch_size):
        indices = []
        for _ in range(batch_size):
            indices.append(self._sample_from_pool())
        return torch.tensor(indices, dtype=torch.long, device=self.device), indices[0]

    def update(self, raw_grad_norm): pass


def custom_collate_fn(batch):
    batch = list(filter(None, batch))
    if not batch: return {}
    output = {}
    for k in batch[0]:
        if k == "original_image": output[k] = [item[k] for item in batch]
        elif isinstance(batch[0][k], torch.Tensor): output[k] = torch.stack([item[k] for item in batch])
        else: output[k] = [item[k] for item in batch]
    return output


def print_dataset_resolution_sample(dataset, sample_count=5):
    sample_count = min(sample_count, len(dataset.items))
    if sample_count <= 0:
        return

    print(f"INFO: Dataset resolution sample ({sample_count} cached item{'s' if sample_count != 1 else ''}):")
    for item in dataset.items[:sample_count]:
        orig_w, orig_h = item["original_size"]
        targ_w, targ_h = item["target_size"]
        orig_ar = orig_w / orig_h if orig_h else 1.0
        targ_ar = targ_w / targ_h if targ_h else 1.0
        ar_error_pct = (abs(orig_ar - targ_ar) / orig_ar * 100) if orig_ar else 0.0
        stem = Path(item["te_path"]).stem.replace("_te", "")
        variant_label = f", variant {item.get('bucket_variant_index', 0)}" if item.get("bucket_variant_index", 0) else ""
        print(
            f"INFO:   {stem}: original {orig_w}x{orig_h} (AR {orig_ar:.4f}) -> "
            f"target {targ_w}x{targ_h} (AR {targ_ar:.4f}){variant_label}, "
            f"AR diff {ar_error_pct:.2f}%, cropped not stretched"
        )


def pack_sdxl_sample_schedule(image_schedule, batch_size):
    packed_schedule = []
    batch_size = max(1, int(batch_size or 1))
    for batch_index, batch in enumerate(image_schedule):
        packed_batch = []
        for local_index, dataset_index in enumerate(batch):
            sample_index = batch_index * batch_size + local_index
            packed_batch.append(ImageTextLatentDataset.pack_sample_index(dataset_index, sample_index))
        packed_schedule.append(packed_batch)
    return packed_schedule


def create_optimizer(config, params_to_optimize):
    optimizer_type = config.OPTIMIZER_TYPE.lower()
    initial_lr = max(p[1] for p in getattr(config, 'LR_CUSTOM_CURVE', [])) if getattr(config, 'LR_CUSTOM_CURVE', []) else config.LEARNING_RATE

    if optimizer_type == "titan":
        final_params = {**default_config.TITAN_PARAMS, **getattr(config, 'TITAN_PARAMS', {})}
        dtype_str = final_params.get('momentum_dtype', 'bfloat16')
        m_dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float32
        return TitanAdamW(params_to_optimize, lr=initial_lr, betas=tuple(final_params.get('betas', [0.9, 0.999])), eps=final_params.get('eps', 1e-8), weight_decay=final_params.get('weight_decay', 0.01), debias_strength=final_params.get('debias_strength', 1.0), momentum_dtype=m_dtype)
    elif optimizer_type == "raven":
        final_params = {**getattr(default_config, 'RAVEN_PARAMS', {}), **getattr(config, 'RAVEN_PARAMS', {})}
        dtype_str = final_params.get('momentum_dtype', 'bfloat16')
        m_dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float32
        return RavenAdamW(params_to_optimize, lr=initial_lr, betas=tuple(final_params.get('betas', [0.9, 0.999])), eps=final_params.get('eps', 1e-8), weight_decay=final_params.get('weight_decay', 0.01), debias_strength=final_params.get('debias_strength', 1.0), momentum_dtype=m_dtype)
    elif optimizer_type == "paged_adamw_8bit":
        try:
            import bitsandbytes as bnb
        except ImportError as exc:
            raise RuntimeError(
                "PagedAdamW8bit requires bitsandbytes, but it is not installed."
            ) from exc
        final_params = {
            **getattr(default_config, 'PAGED_ADAMW_8BIT_PARAMS', {}),
            **getattr(config, 'PAGED_ADAMW_8BIT_PARAMS', {}),
        }
        return bnb.optim.PagedAdamW8bit(
            params_to_optimize,
            lr=initial_lr,
            betas=tuple(final_params.get('betas', [0.9, 0.999])),
            eps=final_params.get('eps', 1e-8),
            weight_decay=final_params.get('weight_decay', 0.01),
            min_8bit_size=4096,
        )
    raise ValueError(f"Unsupported optimizer type: '{config.OPTIMIZER_TYPE}'")


def print_optimizer_summary(optimizer, config):
    optimizer_key = str(config.OPTIMIZER_TYPE).lower()
    group = optimizer.param_groups[0] if optimizer.param_groups else {}
    params = [p for item in optimizer.param_groups for p in item.get("params", [])]
    trainable_tensors = sum(1 for p in params if p.requires_grad)
    trainable_elements = sum(p.numel() for p in params if p.requires_grad)
    names = {
        "raven": "RavenAdamW",
        "paged_adamw_8bit": "PagedAdamW8bit",
        "titan": "TitanAdamW",
    }

    print("\n" + "=" * 58)
    print("INFO: Optimizer Configuration")
    print(f"  - Optimizer:           {names.get(optimizer_key, type(optimizer).__name__)}")
    print(f"  - Config key:          {optimizer_key}")
    print(f"  - Trainable tensors:   {trainable_tensors:,}")
    print(f"  - Trainable elements:  {trainable_elements:,}")
    print(f"  - Initial LR:          {group.get('lr', 0.0):.8g}")
    print(f"  - Betas:               {tuple(group.get('betas', (0.9, 0.999)))}")
    print(f"  - Epsilon:             {group.get('eps', 1e-8):.8g}")
    print(f"  - Weight decay:        {group.get('weight_decay', 0.0):.8g}")

    if optimizer_key == "paged_adamw_8bit":
        print("  - Optimizer state:     paged blockwise 8-bit moments")
        print(f"  - Minimum 8-bit size:  {group.get('min_8bit_size', 4096):,} elements")
        print("  - Bias correction:     standard AdamW")
        print("  - Parameter/grad dtype: unchanged from training configuration")
    else:
        momentum_dtype = group.get("momentum_dtype", getattr(optimizer, "_momentum_dtype", torch.bfloat16))
        dtype_name = str(momentum_dtype).removeprefix("torch.")
        print(f"  - Debias strength:     {group.get('debias_strength', 1.0):.8g}")
        print(f"  - Momentum state:      CPU {dtype_name}")
        print("  - Update math:         FP32 reusable GPU scratch buffer")
        if optimizer_key == "titan":
            print("  - Gradient storage:    pageable CPU FP32 after backward")
        else:
            print("  - Gradient storage:    training device")
    print("=" * 58 + "\n")


def output_model_stem(config, source_path):
    cached = getattr(config, "_RESOLVED_OUTPUT_STEM", None)
    if cached:
        return cached
    requested = str(getattr(config, "OUTPUT_NAME", "auto") or "auto").strip()
    if requested.lower() == "auto":
        requested = f"{Path(source_path).stem}_trained_{{uuid}}"
    run_uuid = "".join(secrets.choice(string.ascii_lowercase + string.digits) for _ in range(6))
    requested = requested.replace("{uuid}", run_uuid)
    requested = Path(requested).name
    if requested.lower().endswith(".safetensors"):
        requested = requested[:-len(".safetensors")]
    requested = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", requested).strip(" .")
    resolved = requested or f"{Path(source_path).stem}_trained_{run_uuid}"
    config._RESOLVED_OUTPUT_STEM = resolved
    return resolved

def bell_timestep_loss_curve(total_timestep_count, device=None, dtype=torch.float32):
    steps = int(total_timestep_count)
    grid = torch.arange(steps, device=device, dtype=dtype)
    y = torch.exp(-2.0 * ((grid - steps / 2) / steps).pow(2))
    y_min = y.min()
    scale = steps / (y - y_min).sum().clamp_min(1e-12)
    return (y - y_min).clamp_min(0.0) * scale


def timestep_loss_curve_from_config(config, total_timestep_count, device=None, dtype=torch.float32):
    steps = int(total_timestep_count)
    if steps <= 0:
        return torch.ones(1, device=device, dtype=dtype)

    points = getattr(config, "TIMESTEP_LOSS_WEIGHT_CURVE", None)
    if not points:
        return torch.ones(steps, device=device, dtype=dtype)
    if isinstance(points, dict):
        preset = str(points.get("preset", "")).lower()
        if preset == "bell":
            return bell_timestep_loss_curve(steps, device=device, dtype=dtype)
        return torch.ones(steps, device=device, dtype=dtype)

    cleaned = []
    for point in points:
        try:
            x = max(0.0, min(1.0, float(point[0])))
            y = max(0.0, float(point[1]))
            cleaned.append((x, y))
        except (TypeError, ValueError, IndexError):
            continue
    if len(cleaned) < 2:
        return torch.ones(steps, device=device, dtype=dtype)

    cleaned.sort(key=lambda p: p[0])
    if cleaned[0][0] > 0.0:
        cleaned.insert(0, (0.0, cleaned[0][1]))
    else:
        cleaned[0] = (0.0, cleaned[0][1])
    if cleaned[-1][0] < 1.0:
        cleaned.append((1.0, cleaned[-1][1]))
    else:
        cleaned[-1] = (1.0, cleaned[-1][1])

    xp = torch.tensor([p[0] for p in cleaned], dtype=torch.float32)
    yp = torch.tensor([p[1] for p in cleaned], dtype=torch.float32)
    grid = torch.linspace(0.0, 1.0, steps, dtype=torch.float32)
    indices = torch.searchsorted(xp, grid, right=True).clamp(1, len(cleaned) - 1)
    x0 = xp[indices - 1]
    x1 = xp[indices]
    y0 = yp[indices - 1]
    y1 = yp[indices]
    blend = ((grid - x0) / (x1 - x0).clamp_min(1e-12)).clamp(0.0, 1.0)
    curve = y0 + (y1 - y0) * blend
    return curve.to(device=device, dtype=dtype)


def weighted_sdxl_mse_loss(pred, target, timesteps, timestep_loss_weights=None):
    per_sample_loss = (pred.float() - target.float()).pow(2).flatten(1).mean(dim=1)
    if timestep_loss_weights is None:
        weights = torch.ones_like(per_sample_loss)
    else:
        curve = timestep_loss_weights.to(device=per_sample_loss.device, dtype=per_sample_loss.dtype)
        indices = timesteps.long().clamp(0, curve.shape[0] - 1)
        weights = curve[indices]
    return (per_sample_loss * weights).mean()

def _get_sdxl_unet_conversion_map():
    unet_conversion_map = [
        ("time_embed.0.weight", "time_embedding.linear_1.weight"), ("time_embed.0.bias", "time_embedding.linear_1.bias"),
        ("time_embed.2.weight", "time_embedding.linear_2.weight"), ("time_embed.2.bias", "time_embedding.linear_2.bias"),
        ("input_blocks.0.0.weight", "conv_in.weight"), ("input_blocks.0.0.bias", "conv_in.bias"),
        ("out.0.weight", "conv_norm_out.weight"), ("out.0.bias", "conv_norm_out.bias"),
        ("out.2.weight", "conv_out.weight"), ("out.2.bias", "conv_out.bias"),
        ("label_emb.0.0.weight", "add_embedding.linear_1.weight"), ("label_emb.0.0.bias", "add_embedding.linear_1.bias"),
        ("label_emb.0.2.weight", "add_embedding.linear_2.weight"), ("label_emb.0.2.bias", "add_embedding.linear_2.bias"),
    ]
    unet_conversion_map_resnet = [
        ("in_layers.0", "norm1"), ("in_layers.2", "conv1"),
        ("out_layers.0", "norm2"), ("out_layers.3", "conv2"),
        ("emb_layers.1", "time_emb_proj"), ("skip_connection", "conv_shortcut"),
    ]
    unet_conversion_map_layer = []
    for i in range(3):
        for j in range(2):
            unet_conversion_map_layer.append((f"input_blocks.{3 * i + j + 1}.0.", f"down_blocks.{i}.resnets.{j}."))
            if i > 0: unet_conversion_map_layer.append((f"input_blocks.{3 * i + j + 1}.1.", f"down_blocks.{i}.attentions.{j}."))
        for j in range(3):
            unet_conversion_map_layer.append((f"output_blocks.{3 * i + j}.0.", f"up_blocks.{i}.resnets.{j}."))
            if i < 2: unet_conversion_map_layer.append((f"output_blocks.{3 * i + j}.1.", f"up_blocks.{i}.attentions.{j}."))
        if i < 3:
            unet_conversion_map_layer.append((f"input_blocks.{3 * (i + 1)}.0.op.", f"down_blocks.{i}.downsamplers.0.conv."))
            unet_conversion_map_layer.append((f"output_blocks.{3 * i + 2}.{1 if i == 0 else 2}.", f"up_blocks.{i}.upsamplers.0."))
    unet_conversion_map_layer.append(("output_blocks.2.2.conv.", "output_blocks.2.1.conv."))
    unet_conversion_map_layer.append(("middle_block.1.", "mid_block.attentions.0."))
    for j in range(2): unet_conversion_map_layer.append((f"middle_block.{2 * j}.", f"mid_block.resnets.{j}."))
    return unet_conversion_map, unet_conversion_map_resnet, unet_conversion_map_layer

def get_unet_key_mapping(current_keys):
    map_static, map_resnet, map_layer = _get_sdxl_unet_conversion_map()
    mapping = {k: k for k in current_keys}
    for sd_name, hf_name in map_static:
        if hf_name in mapping: mapping[hf_name] = sd_name
    for k, v in mapping.items():
        if "resnets" in k:
            for sd_part, hf_part in map_resnet: v = v.replace(hf_part, sd_part)
            mapping[k] = v
    for k, v in mapping.items():
        for sd_part, hf_part in map_layer:
            if hf_part in v: v = v.replace(hf_part, sd_part)
        mapping[k] = v
    final_mapping = {}
    for hf_name, sd_name in mapping.items():
        final_mapping[hf_name] = sd_name if sd_name.startswith("model.diffusion_model.") else f"model.diffusion_model.{sd_name}"
    return final_mapping

def save_model(output_path, unet, base_checkpoint_path, compute_dtype):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nINFO: Saving model to: {output_path.name}")
    
    try: base_tensors = load_file(str(base_checkpoint_path), device="cpu")
    except Exception as e:
        print(f"ERROR: Could not load base checkpoint: {e}")
        return

    key_map = get_unet_key_mapping(list(unet.state_dict().keys()))
    unet_state = unet.state_dict()
    
    print(f"INFO: Base checkpoint keys: {len(base_tensors)}")
    print(f"INFO: UNet keys to merge:   {len(key_map)}")

    converted = 0
    for key in base_tensors.keys():
        if base_tensors[key].dtype in [torch.float32, torch.float16, torch.bfloat16]:
            base_tensors[key] = base_tensors[key].to(dtype=compute_dtype)
            converted += 1
    print(f"INFO: Converted {converted} base tensors to {compute_dtype}")

    updated, missing = 0, []
    for hf_key, target_key in key_map.items():
        if hf_key in unet_state:
            if target_key not in base_tensors:
                missing.append(target_key)
            tensor_cpu = unet_state[hf_key].detach().to("cpu", dtype=compute_dtype)
            base_tensors[target_key] = tensor_cpu
            updated += 1
            del tensor_cpu

    print(f"INFO: Merged {updated} UNet layers into checkpoint")
    if missing:
        print(f"WARNING: {len(missing)} keys not found in base checkpoint (new keys added):")
        for k in missing[:5]: print(f"  -> {k}")
        if len(missing) > 5: print(f"  ... and {len(missing) - 5} more")

    print(f"INFO: Saving to disk...")
    save_file(base_tensors, str(output_path))
    print(f"INFO: Save complete -> {output_path.name}")

    del base_tensors, unet_state, key_map
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

def save_checkpoint_pt(global_step, micro_step, unet, base_checkpoint_path, optimizer, lr_scheduler, scaler, sampler, config, timestep_sampler=None):
    output_dir = Path(config.OUTPUT_DIR)
    output_stem = output_model_stem(config, config.SINGLE_FILE_CHECKPOINT_PATH)
    model_filename = f"{output_stem}_step_{global_step}.safetensors"
    state_filename = f"{output_stem}_training_state_step_{global_step}.pt"
    save_model(output_dir / model_filename, unet, base_checkpoint_path, config.compute_dtype)

    optim_state = optimizer.save_cpu_state() if hasattr(optimizer, 'save_cpu_state') else optimizer.state_dict()
    training_state = {
        'global_step': global_step, 'micro_step': micro_step, 'optimizer_state': optim_state,
        'sampler_seed': sampler.seed, 'sampler_epoch': max(sampler.epoch - 1, 0),
        'timestep_sampler_state': timestep_sampler.state_dict() if timestep_sampler is not None and hasattr(timestep_sampler, "state_dict") else None,
        'random_state': random.getstate(), 'numpy_state': np.random.get_state(),
        'torch_cpu_state': torch.get_rng_state(),
        'torch_cuda_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }
    torch.save(training_state, output_dir / state_filename)


def consume_force_save_flag(flag_path):
    if not flag_path.exists():
        return False
    try:
        flag_path.unlink()
        return True
    except OSError as e:
        print(f"WARNING: Emergency checkpoint flag found but could not be deleted: {e}")
        return False


def main():
    config = TrainingConfig()
    if config.SEED: set_seed(config.SEED)

    OUTPUT_DIR = Path(config.OUTPUT_DIR)
    force_save_flag = Path(__file__).resolve().with_name("force_save.flag")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    global_step, micro_step, optimizer_step = 0, 0, 0
    model_to_load = Path(config.SINGLE_FILE_CHECKPOINT_PATH)
    initial_sampler_seed, optimizer_state, initial_epoch = config.SEED, None, 0
    initial_timestep_sampler_state = None

    if config.RESUME_TRAINING:
        print("\n" + "="*50 + "\n--- RESUMING TRAINING SESSION ---\n")
        training_state = torch.load(Path(config.RESUME_STATE_PATH), map_location="cpu", weights_only=False)
        global_step_saved   = training_state.get('global_step', 0)
        micro_step          = training_state.get('micro_step', global_step_saved * config.GRADIENT_ACCUMULATION_STEPS)
        optimizer_step      = micro_step // config.GRADIENT_ACCUMULATION_STEPS
        initial_sampler_seed = training_state['sampler_seed']
        initial_epoch       = training_state.get('sampler_epoch', 0)
        initial_timestep_sampler_state = training_state.get('timestep_sampler_state')
        optimizer_state     = training_state['optimizer_state']
        model_to_load       = Path(config.RESUME_MODEL_PATH)
        if 'random_state'    in training_state: random.setstate(training_state['random_state'])
        if 'numpy_state'     in training_state: np.random.set_state(training_state['numpy_state'])
        if 'torch_cpu_state' in training_state: torch.set_rng_state(training_state['torch_cpu_state'])
        if 'torch_cuda_state' in training_state and training_state['torch_cuda_state'] is not None:
            torch.cuda.set_rng_state(training_state['torch_cuda_state'])
    else:
        mode_str = "RECTIFIED FLOW" if config.is_rectified_flow else "STANDARD SDXL"
        print("\n" + "="*50 + f"\n--- STARTING {mode_str} TRAINING ---\n" + "="*50 + "\n")

    print(f"INFO: Noise type: {getattr(config, 'NOISE_MODE', 'normal')}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if check_if_caching_needed(config):
        target_channels = getattr(config, 'VAE_LATENT_CHANNELS', 4)
        conf_shift      = getattr(config, 'VAE_SHIFT_FACTOR', None)
        conf_scale      = getattr(config, 'VAE_SCALING_FACTOR', None)

        vae_source = config.VAE_PATH if (config.VAE_PATH and Path(config.VAE_PATH).exists()) else config.SINGLE_FILE_CHECKPOINT_PATH
        vae_for_caching = load_vae_robust(vae_source, device, target_channels=target_channels)
        vae_for_caching.config.shift_factor   = conf_shift  if conf_shift  is not None else getattr(vae_for_caching.config, 'shift_factor',   None)
        vae_for_caching.config.scaling_factor = conf_scale  if conf_scale  is not None else getattr(vae_for_caching.config, 'scaling_factor', None)
        print(f"INFO: VAE shift={vae_for_caching.config.shift_factor}, scale={vae_for_caching.config.scaling_factor}, channels={vae_for_caching.config.latent_channels}")
        vae_for_caching.enable_tiling()
        vae_for_caching.enable_slicing()

        base_pipe = StableDiffusionXLPipeline.from_single_file(
            config.SINGLE_FILE_CHECKPOINT_PATH, vae=vae_for_caching, unet=None,
            torch_dtype=torch.float32, low_cpu_mem_usage=True
        )
        precompute_and_cache_latents(config, base_pipe.tokenizer, base_pipe.tokenizer_2,
                                     base_pipe.text_encoder, base_pipe.text_encoder_2,
                                     vae_for_caching, device)
        del base_pipe, vae_for_caching
        gc.collect(); torch.cuda.empty_cache()

    print(f"\n--- Loading Model ---")
    unet = load_unet_robust(model_to_load, config.compute_dtype)
    gc.collect(); torch.cuda.empty_cache()

    scheduler = None
    if not config.is_rectified_flow:
        prediction_type = getattr(config, "PREDICTION_TYPE", "epsilon")
        print(f"\n--- Using Standard SDXL ({prediction_type}) ---")
        scheduler = DDPMScheduler.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            subfolder="scheduler"
        )
        scheduler.config.prediction_type = (
            prediction_type if prediction_type == "v_prediction" else "epsilon"
        )
    else:
        print(f"\n--- Using Rectified Flow ---")

    # A common 0-1 telemetry value for the GUI. For DDPM this is the bounded
    # noise coefficient used by add_noise; rectified flow reports its mix t.
    scheduler_noise_sigmas = (
        (1.0 - scheduler.alphas_cumprod.to(device=device, dtype=torch.float32)).clamp_min(0.0).sqrt()
        if scheduler is not None else None
    )

    print("\n--- Initializing Dataset ---")
    dataset   = ImageTextLatentDataset(config)
    print_dataset_resolution_sample(dataset, sample_count=5)
    timestep_sampler = TimestepSampler(config, device)
    if initial_timestep_sampler_state is not None:
        timestep_sampler.load_state_dict(initial_timestep_sampler_state)
    elif config.RESUME_TRAINING and micro_step > 0:
        timestep_sampler.set_current_step(micro_step)
    timestep_loss_weights = timestep_loss_curve_from_config(
        config,
        1000,
        device=device,
        dtype=torch.float32,
    )

    image_schedule = build_image_batch_schedule(
        dataset,
        config.MAX_TRAIN_STEPS,
        config.BATCH_SIZE,
        initial_sampler_seed,
        timestep_sampler.ticket_pool,
        timestep_sampler.bin_ranges,
        bool(getattr(config, "TIMESTEP_FORCE_IMAGE_BIN_SPREAD", False)),
    )
    image_schedule = pack_sdxl_sample_schedule(image_schedule, config.BATCH_SIZE)
    sampler = PrecomputedImageBatchSampler(image_schedule, initial_sampler_seed, micro_step if config.RESUME_TRAINING else 0)
    print(f"INFO: Precomputed image batch schedule for {len(image_schedule):,} step(s).")
    print("INFO: SDXL dataset conditioning is keyed by seed and absolute sample position.")
    dataloader = DataLoader(dataset, batch_sampler=sampler, collate_fn=custom_collate_fn, num_workers=config.NUM_WORKERS)

    unet.enable_gradient_checkpointing()
    unet.to(device)
    set_attention_processor(unet, getattr(config, "MEMORY_EFFICIENT_ATTENTION", "sdpa"))

    exclusion_keywords = getattr(config, "UNET_EXCLUDE_TARGETS", [])
    for name, param in unet.named_parameters():
        should_exclude = any(fnmatch.fnmatch(name, kw if '*' in kw else f"*{kw}*") for kw in exclusion_keywords)
        param.requires_grad = not should_exclude

    total_params     = sum(p.numel() for p in unet.parameters())
    frozen_params    = sum(p.numel() for p in unet.parameters() if not p.requires_grad)
    trainable_params = total_params - frozen_params
    print(f"\n{'='*50}\nINFO: UNet Parameter Statistics:")
    print(f"  - Total Parameters:     {total_params:,}")
    print(f"  - Frozen Parameters:    {frozen_params:,}")
    print(f"  - Trainable Parameters: {trainable_params:,}")
    print(f"  - Percentage Frozen:    {(frozen_params/total_params)*100:.2f}%")
    print("="*50 + "\n")

    params_to_optimize = [{"params": [p for p in unet.parameters() if p.requires_grad], "lr_scale": 1.0}]
    optimizer    = create_optimizer(config, params_to_optimize)
    lr_scheduler = CustomCurveLRScheduler(optimizer=optimizer, curve_points=config.LR_CUSTOM_CURVE, total_micro_steps=config.MAX_TRAIN_STEPS)
    all_trainable_params = [p for g in params_to_optimize for p in g['params']]

    if config.RESUME_TRAINING:
        if optimizer_state:
            try:
                if hasattr(optimizer, 'load_cpu_state'):
                    optimizer.load_cpu_state(optimizer_state)
                else:
                    optimizer.load_state_dict(optimizer_state)
            except Exception as e:
                raise RuntimeError(
                    "Failed to restore the optimizer state while resuming SDXL training. "
                    "The resume was aborted to avoid continuing with a fresh optimizer."
                ) from e
        lr_scheduler.step(micro_step)

    print_optimizer_summary(optimizer, config)

    unet.train()
    diagnostics  = TrainingDiagnostics(config.GRADIENT_ACCUMULATION_STEPS)
    reporter     = AsyncReporter(total_steps=config.MAX_TRAIN_STEPS, test_param_name="conv_in" if hasattr(unet, 'conv_in') else "first param")
    
    accumulated_latent_paths  = []
    global_step_times         = deque(maxlen=50)
    optim_step_times          = deque(maxlen=20)
    training_start_time       = time.time()
    last_step_time            = time.time()
    last_optim_step_log_time  = time.time()
    done                      = False
    dataloader_len            = len(dataloader)
    if config.RESUME_TRAINING and dataloader_len <= 0:
        reporter.log_message("WARNING: Resume requested but dataloader is empty; starting without batch skipping.")
    noise_generator = torch.Generator(device=device)

    while not done:
        for batch in dataloader:
            if micro_step >= config.MAX_TRAIN_STEPS: done = True; break
            if not batch: continue

            micro_step += 1
            diag_data_to_log = None 

            if "latent_path" in batch:
                accumulated_latent_paths.extend(batch["latent_path"])

            latents = batch["latents"].to(device, non_blocking=True).detach()
            embeds  = batch["embeds"].to(device, non_blocking=True, dtype=config.compute_dtype)
            pooled  = batch["pooled"].to(device, non_blocking=True, dtype=config.compute_dtype)
            batch_crop_coords = batch.get("crop_coords", [(0, 0)] * len(batch["latents"]))

            with torch.autocast(device_type=device.type, dtype=config.compute_dtype, enabled=True):

                scaled_sizes = batch.get("scaled_sizes", batch["original_sizes"])
                time_ids_data = [
                    [s1[1], s1[0], crop[0], crop[1], s2[1], s2[0]]
                    for s1, crop, s2 in zip(scaled_sizes, batch_crop_coords, batch["target_sizes"])
                ]
                time_ids = torch.tensor(time_ids_data, dtype=config.compute_dtype, device=device)

                timesteps, first_timestep_int = timestep_sampler.sample(latents.shape[0])
                timestep_str = str(first_timestep_int)
                noise = generate_noise(
                    latents=latents,
                    generator=noise_generator,
                    device=device,
                    dtype=config.compute_dtype,
                    step=micro_step,
                    seed=config.SEED
                )
                if config.is_rectified_flow:
                    jitter_generator = seeded_torch_generator(device, config.SEED, micro_step, 0x5D1)
                    jitter = torch.rand(timesteps.shape, device=device, dtype=torch.float32, generator=jitter_generator)
                    t_continuous = ((timesteps.float() + jitter) / 1000.0).clamp(0.0, 1.0)

                    t_exp = t_continuous.view(-1, 1, 1, 1)
                    reported_sigmas = t_continuous
                    noisy_latents = (1 - t_exp) * latents + t_exp * noise
                    target = noise - latents
                    timesteps_conditioning = (t_continuous * 1000.0)
                else:
                    reported_sigmas = scheduler_noise_sigmas[timesteps.long()]
                    noisy_latents = scheduler.add_noise(latents, noise, timesteps)
                    target = (scheduler.get_velocity(latents, noise, timesteps)
                              if scheduler.config.prediction_type == "v_prediction" else noise)
                    timesteps_conditioning = timesteps

                pred = unet(noisy_latents.to(config.compute_dtype), timesteps_conditioning, embeds,
                            added_cond_kwargs={"text_embeds": pooled, "time_ids": time_ids}).sample

                loss = weighted_sdxl_mse_loss(pred, target, timesteps, timestep_loss_weights)

                (loss / config.GRADIENT_ACCUMULATION_STEPS).backward()

            raw_loss_value = loss.detach().item()
            diagnostics.step(raw_loss_value)
            lr_scheduler.step(micro_step)

            if micro_step % config.GRADIENT_ACCUMULATION_STEPS == 0:
                raw_grad_norm = (
                    optimizer.clip_grad_norm(config.CLIP_GRAD_NORM if config.CLIP_GRAD_NORM > 0 else float('inf'))
                    if isinstance(optimizer, TitanAdamW)
                    else torch.nn.utils.clip_grad_norm_(
                        all_trainable_params,
                        config.CLIP_GRAD_NORM if config.CLIP_GRAD_NORM > 0 else float('inf')
                    )
                )
                if isinstance(raw_grad_norm, torch.Tensor): raw_grad_norm = raw_grad_norm.item()
                clipped_grad_norm = min(raw_grad_norm, config.CLIP_GRAD_NORM) if config.CLIP_GRAD_NORM > 0 else raw_grad_norm

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1

                optim_step_time = time.time() - last_optim_step_log_time
                optim_step_times.append(optim_step_time)
                last_optim_step_log_time = time.time()

                diag_data_to_log = {
                    'optim_step':          optimizer_step,
                    'avg_loss':            diagnostics.get_average_loss(),
                    'current_lr':          optimizer.param_groups[-1]['lr'],
                    'raw_grad_norm':       raw_grad_norm,
                    'clipped_grad_norm':   clipped_grad_norm,
                    'update_delta':        1.0 if raw_grad_norm > 0 else 0.0,
                    'optim_step_time':     optim_step_time,
                    'avg_optim_step_time': sum(optim_step_times) / len(optim_step_times),
                }

                diagnostics.reset()
                accumulated_latent_paths.clear()

                scheduled_save = (
                    config.SAVE_EVERY_N_STEPS > 0 and
                    optimizer_step > 0 and
                    optimizer_step % config.SAVE_EVERY_N_STEPS == 0
                )
                force_save = consume_force_save_flag(force_save_flag)
                if scheduled_save or force_save:
                    reason = "Emergency checkpoint requested" if force_save and not scheduled_save else "Saving checkpoint"
                    reporter.log_message(f"\n--- {reason} at optimizer step {optimizer_step} ---")
                    save_checkpoint_pt(optimizer_step, micro_step, unet, model_to_load,
                                       optimizer, lr_scheduler, None, sampler, config, timestep_sampler)

            step_duration = time.time() - last_step_time
            global_step_times.append(step_duration)
            last_step_time = time.time()

            reporter.log_step(micro_step, timing_data={
                'raw_step_time': step_duration,
                'elapsed_time':  time.time() - training_start_time,
                'eta':           (config.MAX_TRAIN_STEPS - micro_step) * (sum(global_step_times) / len(global_step_times)) if global_step_times else 0,
                'loss':          raw_loss_value,
                'timestep':      timestep_str,
                'sigma':         float(reported_sigmas[0].detach().float().item()),
            }, diag_data=diag_data_to_log)

    reporter.log_message("\nTraining complete.")
    reporter.shutdown()
    save_model(
        OUTPUT_DIR / f"{output_model_stem(config, config.SINGLE_FILE_CHECKPOINT_PATH)}.safetensors",
        unet, model_to_load, config.compute_dtype
    )
    print("All tasks complete. Final model saved.")


if __name__ == "__main__":
    try: multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError: pass
    main()
