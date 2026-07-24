import gc
import fnmatch
import hashlib
import json
import math
import os
import random
import re
import struct
import sys
import time
import threading
import traceback
import uuid
from collections import defaultdict, deque
from multiprocessing import Pool, cpu_count
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors import safe_open
from safetensors.torch import save_file
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from scripts.semantic import generate_lineart_loss_map
from training_utils.caching.cache import (
    CAPTION_JSON_PRIMARY_TYPE,
    CACHE_IMAGE_LAYOUT_OPTION_KEYS,
    cache_image_layout_options_match,
    cache_index_exists,
    cache_latent_options_match,
    cache_payload_options_match_for_index,
    cache_payload_options_match_for_index_item,
    cache_stem_for_image,
    cache_text_options_match,
    cached_file_signatures_match,
    caption_file_signature_for_image,
    caption_source_type,
    caption_types_for_cache,
    collect_image_paths,
    image_file_signature,
    json_caption_cache_suffix,
    json_caption_mode_enabled,
    load_cache_index,
    remove_cache_files_for_stem,
    save_cache_index,
    selected_caption_variant_path,
    strip_json_caption_suffix,
    text_cache_paths_for_index_item,
)


from train import (
    AsyncReporter,
    BucketBatchSampler,
    CustomCurveLRScheduler,
    PrecomputedImageBatchSampler,
    TitanAdamW,
    TrainingDiagnostics,
    build_timestep_ticket_pool,
    build_image_batch_schedule,
    caption_signature_for_image,
    consume_force_save_flag,
    create_optimizer,
    fix_alpha_channel,
    get_json_caption_weights,
    get_max_bucket_resolution_for_config,
    get_multi_bucket_resolutions,
    get_vae_source_for_config,
    get_text_conditioning_scale_range,
    make_bucket_variant_metadata,
    null_conditioning_cache_needed,
    set_seed,
    smart_resize,
    timestep_loss_curve_from_config,
    text_cache_float_dtype,
    text_cache_float_dtype_name,
    validate_and_assign_resolution,
    vae_cache_float_dtype,
    vae_cache_float_dtype_name,
    BUCKET_LAYOUT_VERSION,
)

AnimaImagePipeline = None
ModelConfig = None

# Workaround for Comfy quantized DiT weights: dequantize to BF16 for local Anima computation.
ANIMA_QAT_TARGET_FORMAT = "bfp16"  # bf16/bfp16, nvfp4, int8, e4m3, or e5m2
ANIMA_QAT_NVFP4_SCALE_MULTIPLIER = 1.0
ANIMA_REPAIR_LINEART_LOSS_ENABLED = True
ANIMA_REPAIR_LINEART_LOSS_STRENGTH = 1.0


def configure_console_output():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(errors="replace")
            except Exception:
                pass


configure_console_output()


def normalize_anima_config(config):
    if getattr(config, "DIT_VAE_PATH", ""):
        config.VAE_PATH = config.DIT_VAE_PATH
    exclude_targets = getattr(config, "DIT_EXCLUDE_TARGETS", [])
    if isinstance(exclude_targets, str):
        config.DIT_EXCLUDE_TARGETS = [item.strip() for item in exclude_targets.split(",") if item.strip()]
    elif isinstance(exclude_targets, list):
        config.DIT_EXCLUDE_TARGETS = [item for item in exclude_targets if item]
    else:
        config.DIT_EXCLUDE_TARGETS = []


def anima_cache_folder_name(config):
    return getattr(config, "ANIMA_CACHE_FOLDER_NAME", ".precomputed_anima_dit_cache")


def anima_text_cache_float_dtype(config):
    return text_cache_float_dtype(config)


def anima_text_cache_float_dtype_name(config):
    return text_cache_float_dtype_name(config)


def anima_vae_cache_float_dtype(config):
    return vae_cache_float_dtype(config)


def anima_vae_cache_float_dtype_name(config):
    return vae_cache_float_dtype_name(config)


def anima_caption_types_for_cache(config, json_caption_mode=None):
    if json_caption_mode is None:
        json_caption_mode = json_caption_mode_enabled(config)
    if not json_caption_mode:
        return ("txt",)
    return caption_types_for_cache(True)


def anima_caption_variant_index(variant_paths):
    return {
        caption_type: {"te_path": str(path)}
        for caption_type, path in variant_paths.items()
    }


def anima_dataset_roots(config):
    return [
        Path(ds["path"])
        for ds in getattr(config, "INSTANCE_DATASETS", [])
        if ds.get("path")
    ]


def collect_anima_image_paths(root):
    return collect_image_paths(root)


def anima_cache_stem_for_image(root, image_path, cache_suffix=""):
    return cache_stem_for_image(root, image_path) + cache_suffix


def anima_cache_base_stem(cache_path):
    stem = Path(cache_path).stem
    stem = re.sub(r"_(te|lat)$", "", stem)
    stem = strip_json_caption_suffix(stem)
    return re.sub(r"_mb\d+$", "", stem)


def anima_text_cache_paths_for_index_item(item):
    return text_cache_paths_for_index_item(
        item,
        path_key="te_path",
        primary_key="te_path",
    )


def anima_cache_paths_for_index_item(item):
    paths = list(anima_text_cache_paths_for_index_item(item))
    if item.get("lat_path"):
        paths.append(item["lat_path"])
    return paths


def anima_lat_path_for_item(item):
    return item.get("lat_path")


def remove_anima_cache_file(cache_path):
    try:
        cache_path = Path(cache_path)
        if cache_path.exists():
            cache_path.unlink()
    except OSError as e:
        print(f"WARNING: Could not remove stale Anima cache file {cache_path}: {e}")


def anima_expected_cache_paths(root, cache_dir, meta, active_caption_types, json_caption_mode):
    safe_filename = anima_cache_stem_for_image(root, meta["ip"], meta.get("cache_suffix", ""))
    text_paths = {
        caption_type: cache_dir / f"{safe_filename}{json_caption_cache_suffix(caption_type, json_caption_mode)}_te.pt"
        for caption_type in active_caption_types
    }
    return text_paths, cache_dir / f"{safe_filename}_lat.pt"


def anima_metadata_matches(payload, meta, check_caption=True):
    if not isinstance(payload, dict):
        return False
    if check_caption and payload.get("caption_signature") != meta.get("caption_signature"):
        return False
    relative_path = str(meta["ip"].relative_to(meta["root"]))
    return (
        payload.get("relative_path") == relative_path
        and tuple(payload.get("original_size", ())) == tuple(meta["original_size"])
        and tuple(payload.get("scaled_size", ())) == tuple(meta["scaled_size"])
        and tuple(payload.get("target_size", ())) == tuple(meta["target_resolution"])
        and tuple(payload.get("crop_coords", (0, 0))) == tuple(meta.get("crop_coords", (0, 0)))
        and int(payload.get("bucket_variant_index", 0) or 0) == int(meta.get("bucket_variant_index", 0) or 0)
    )


def anima_text_cache_valid(path, meta, caption_type, caption, text_cache_dtype, expected_options):
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        prompt_emb = payload.get("prompt_emb")
        t5xxl_ids = payload.get("t5xxl_ids")
        return (
            prompt_emb is not None
            and t5xxl_ids is not None
            and prompt_emb.dtype == text_cache_dtype
            and payload.get("caption_type") == caption_type
            and payload.get("caption") == caption
            and anima_metadata_matches(payload, meta, check_caption=True)
            and anima_text_cache_compatible_options(payload.get("cache_options"), expected_options)
        )
    except Exception:
        return False


def anima_lat_cache_valid(path, meta, vae_cache_dtype, expected_options):
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        latents = payload.get("latents") if isinstance(payload, dict) else None
        return (
            latents is not None
            and latents.dtype == vae_cache_dtype
            and not torch.isnan(latents).any()
            and not torch.isinf(latents).any()
            and anima_metadata_matches(payload, meta, check_caption=False)
            and anima_lat_cache_compatible_options(payload.get("cache_options"), expected_options)
        )
    except Exception:
        return False


def get_anima_cache_options(config):
    multi_bucket_enabled = bool(getattr(config, "MULTI_BUCKET_ENABLED", False))
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
        "version": 6,
        "cache_schema_version": 1,
        "bucket_layout": BUCKET_LAYOUT_VERSION,
        "text_cache_float_dtype": anima_text_cache_float_dtype_name(config),
        "vae_cache_float_dtype": anima_vae_cache_float_dtype_name(config),
        "caption_source_type": caption_source_type(config),
        "caption_json_types": list(anima_caption_types_for_cache(config)),
        "caption_chunking_enabled": False,
        "caption_embedding_layout": "anima_qwen_t5_ids",
        "max_bucket_resolution": get_max_bucket_resolution_for_config(config),
        "should_upscale": bool(getattr(config, "SHOULD_UPSCALE", False)),
        "multi_bucket_enabled": multi_bucket_enabled,
        "multi_bucket_extra_buckets": (
            int(getattr(config, "MULTI_BUCKET_EXTRA_BUCKETS", 0) or 0)
            if multi_bucket_enabled
            else 0
        ),
        "vae_normalization_mode": getattr(config, "VAE_NORMALIZATION_MODE", "scalar"),
        "vae_shift_factor": getattr(config, "VAE_SHIFT_FACTOR", None),
        "vae_scaling_factor": getattr(config, "VAE_SCALING_FACTOR", None),
        "vae_latent_channels": getattr(config, "VAE_LATENT_CHANNELS", None),
        "vae_path": str(getattr(config, "VAE_PATH", "") or ""),
        "vae_source_path": vae_source_path,
        "vae_source_size": vae_source_size,
        "vae_source_mtime_ns": vae_source_mtime_ns,
        "vae_caching_tiled": bool(getattr(config, "VAE_CACHING_TILED", True)),
        "vae_caching_tile_size": list(getattr(config, "VAE_CACHING_TILE_SIZE", [96, 96])),
        "vae_caching_tile_stride": list(getattr(config, "VAE_CACHING_TILE_STRIDE", [72, 72])),
        "repair_lineart_loss_enabled": ANIMA_REPAIR_LINEART_LOSS_ENABLED,
        "repair_lineart_mask_version": 6,
    }


def anima_image_layout_options_match(cached_options, expected_options):
    return cache_image_layout_options_match(
        cached_options,
        expected_options,
        extra_keys=("caption_json_types", "repair_lineart_loss_enabled", "repair_lineart_mask_version"),
    )


def anima_text_cache_compatible_options(cached_options, expected_options):
    return cache_text_options_match(
        cached_options,
        expected_options,
    )


def anima_lat_cache_compatible_options(cached_options, expected_options):
    return cache_latent_options_match(
        cached_options,
        expected_options,
        extra_keys=(
            "vae_caching_tiled",
            "vae_caching_tile_size",
            "vae_caching_tile_stride",
            "repair_lineart_loss_enabled",
            "repair_lineart_mask_version",
        ),
    )


def anima_cache_rebuild_needed_for_root(config, root, expected_options=None, cache_name=None):
    expected_options = expected_options or get_anima_cache_options(config)
    cache_name = cache_name or anima_cache_folder_name(config)
    cache_dir = root / cache_name
    if not cache_dir.exists() or not cache_index_exists(cache_dir):
        print(
            "INFO: Anima cache rebuild needed for "
            f"{root}: cache_dir_exists={cache_dir.exists()}, "
            f"index_exists={cache_index_exists(cache_dir)}."
        )
        return True
    try:
        index_data = load_cache_index(cache_dir)
        image_paths = collect_anima_image_paths(root)
        current_cache_stems = {anima_cache_stem_for_image(root, p) for p in image_paths}
        cached_options = index_data.get("cache_options")
        if not anima_image_layout_options_match(cached_options, expected_options):
            print(f"INFO: Anima cache rebuild needed for {root}: cache options changed.")
            if isinstance(cached_options, dict):
                for key in CACHE_IMAGE_LAYOUT_OPTION_KEYS + ("caption_json_types",):
                    cached_value = cached_options.get(key, "<missing>")
                    expected_value = expected_options.get(key, "<missing>")
                    if cached_value != expected_value:
                        print(f"  - {key}: cached={cached_value!r}, expected={expected_value!r}")
            else:
                print(f"  cached_options={cached_options!r}")
                print(f"  expected_options={expected_options!r}")
            return True
        files = index_data.get("files", [])
        if not files:
            print(f"INFO: Anima cache rebuild needed for {root}: dataset_index has no files.")
            return True
        indexed_base_stems = {
            anima_cache_base_stem(cache_path)
            for item in files
            for cache_path in anima_cache_paths_for_index_item(item)
        }
        if not current_cache_stems.issubset(indexed_base_stems):
            print(f"INFO: Anima cache rebuild needed for {root}: new image(s) are not cached.")
            return True
        if any(stem not in current_cache_stems for stem in indexed_base_stems):
            print(f"INFO: Anima cache rebuild needed for {root}: cached image(s) were removed from the dataset.")
            return True
        for item in files:
            cache_paths = anima_cache_paths_for_index_item(item)
            if not cache_paths:
                print(f"INFO: Anima cache rebuild needed for {root}: missing cached path in dataset index.")
                return True
            for cache_path in cache_paths:
                cache_path = Path(cache_path)
                if not cache_path.exists():
                    print(f"INFO: Anima cache rebuild needed for {root}: missing cached item {cache_path}.")
                    return True
            if bool(getattr(config, "ANIMA_DEEP_CACHE_VALIDATE", False)):
                if not cache_payload_options_match_for_index_item(
                    item,
                    expected_options,
                    anima_cache_paths_for_index_item,
                    anima_text_cache_compatible_options,
                    anima_lat_cache_compatible_options,
                ):
                    print(f"INFO: Anima cache rebuild needed for {root}: cache payload options changed.")
                    return True
            relative_path = item.get("relative_path")
            if relative_path:
                try:
                    image_path = root / relative_path
                    stat_match = cached_file_signatures_match(item, image_path, caption_source_type(config))
                    if stat_match is False:
                        print(f"INFO: Anima cache rebuild needed for {root}: image/caption file changed for {relative_path}.")
                        return True
                    if stat_match is None and caption_signature_for_image(image_path, caption_source_type(config)) != item.get("caption_signature"):
                        print(f"INFO: Anima cache rebuild needed for {root}: caption changed for {relative_path}.")
                        return True
                except Exception as e:
                    print(f"INFO: Anima cache rebuild needed for {root}: caption invalid for {relative_path}: {e}")
                    return True
    except Exception:
        print(f"INFO: Anima cache rebuild needed for {root}: failed to read/validate cache index.")
        traceback.print_exc()
        return True
    return False


def anima_roots_needing_cache_rebuild(config):
    roots = anima_dataset_roots(config)
    if bool(getattr(config, "REBUILD_CACHE", False)):
        print("INFO: Rebuilding Anima DiT cache because REBUILD_CACHE=True.")
        return roots

    expected_options = get_anima_cache_options(config)
    cache_name = anima_cache_folder_name(config)
    return [
        root
        for root in roots
        if anima_cache_rebuild_needed_for_root(config, root, expected_options, cache_name)
    ]


def check_if_anima_caching_needed(config):
    return bool(anima_roots_needing_cache_rebuild(config))


def default_huggingface_cache_root():
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return str(Path(hf_home).expanduser())

    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache_home:
        return str(Path(xdg_cache_home).expanduser() / "huggingface")

    return str(Path.home() / ".cache" / "huggingface")


def local_tokenizer_path(model_id, origin_file_pattern):
    origin = (origin_file_pattern or "./").rstrip("/")
    relative_origin = "" if origin in ("", ".") else origin
    candidates = []

    model_path = Path(model_id).expanduser()
    if model_path.exists():
        candidates.append(model_path / relative_origin if relative_origin else model_path)

    for base in (Path("models"), Path(default_huggingface_cache_root())):
        candidates.append(base / model_id / relative_origin if relative_origin else base / model_id)

    required_files = ("tokenizer.json", "spiece.model", "tokenizer.model", "vocab.json")
    for candidate in candidates:
        if candidate.exists() and any((candidate / name).exists() for name in required_files):
            return str(candidate)
    return None


def tokenizer_folder_has_required_files(path):
    if not path:
        return False
    folder = Path(path).expanduser()
    if folder.is_file():
        folder = folder.parent
    required_files = ("tokenizer.json", "spiece.model", "tokenizer.model", "vocab.json")
    return folder.exists() and any((folder / name).exists() for name in required_files)


def resolve_local_tokenizer_path(value, label, default_model_id=None, default_origin="./"):
    raw_value = str(value or "").strip()
    candidates = []

    if raw_value:
        direct = Path(raw_value).expanduser()
        candidates.append(direct.parent if direct.is_file() else direct)

    if default_model_id:
        default_path = local_tokenizer_path(default_model_id, default_origin)
        if default_path:
            candidates.append(Path(default_path))

    for candidate in candidates:
        if tokenizer_folder_has_required_files(candidate):
            return str(candidate)

    raise FileNotFoundError(
        f"{label} tokenizer folder is required for Anima DiT training. "
        "Select a local folder containing tokenizer.json, spiece.model, tokenizer.model, or vocab.json. "
        "Use the tokenizer help button in the GUI for download links."
    )


def make_anima_tokenizer_config(path, label):
    local_path = resolve_local_tokenizer_path(path, label)
    return ModelConfig(path=local_path, skip_download=True)


def ensure_training_utils_available():
    global AnimaImagePipeline, ModelConfig
    if AnimaImagePipeline is not None and ModelConfig is not None:
        return
    try:
        from training_utils.anima import AnimaImagePipeline as _AnimaImagePipeline
        from training_utils.anima import ModelConfig as _ModelConfig
    except Exception as e:
        raise RuntimeError(
            "Anima DiT repair training requires the bundled training_utils package."
        ) from e
    AnimaImagePipeline = _AnimaImagePipeline
    ModelConfig = _ModelConfig


def detect_anima_dit_key_prefix(path):
    prefixes = ("pipe.dit.", "dit.", "model.diffusion_model.", "diffusion_model.", "model.", "net.")
    counts = {prefix: 0 for prefix in prefixes}
    total = 0
    with safe_open(str(path), framework="pt", device="cpu") as f:
        for key in f.keys():
            total += 1
            for prefix in prefixes:
                if key.startswith(prefix):
                    counts[prefix] += 1
                    break
    if total <= 0:
        return ""
    best_prefix, best_count = max(counts.items(), key=lambda item: item[1])
    return best_prefix if best_count / total >= 0.8 else ""



def _anima_checkpoint_has_comfy_quant(path):
    try:
        with safe_open(str(path), framework="pt", device="cpu") as f:
            return any(key.endswith(".comfy_quant") for key in f.keys())
    except Exception:
        return False


def _quant_payload(tensor):
    return json.loads(bytes(tensor.cpu().to(torch.uint8).tolist()).decode("utf-8"))


def _prepare_quantized_training_view(config, quant_path):
    """Dequantize supported Comfy quantized weights for local Anima computation."""
    if not _anima_checkpoint_has_comfy_quant(quant_path):
        raise ValueError("The experimental Anima repair trainer expects a Comfy-quantized DIT_PATH/resume checkpoint.")
    from scripts.convert_anima_to_quants import comfy_quant_key_for_weight, comfy_scale2_key_for_weight, comfy_scale_key_for_weight, dequantize_nvfp4_tensor, write_streaming_safetensors
    quant_path = Path(quant_path)
    view_dir = Path(config.OUTPUT_DIR) / ".aozora_projected_quant_view"
    view_dir.mkdir(parents=True, exist_ok=True)
    view_path = view_dir / f"{quant_path.stem}_compute_bfloat16.safetensors"
    if not view_path.exists() or view_path.stat().st_mtime < quant_path.stat().st_mtime:
        print("INFO: Building internal BF16 computation view of the selected quantized model.")
        records = []
        with safe_open(str(quant_path), framework="pt", device="cpu") as f:
            keys = set(f.keys())
            metadata = dict(f.metadata() or {})
            sidecars = {k for k in keys if k.endswith((".comfy_quant", ".weight_scale", ".weight_scale_2"))}
            for key in sorted(keys - sidecars):
                tensor = f.get_tensor(key)
                qkey = comfy_quant_key_for_weight(key)
                if key.endswith(".weight") and qkey in keys:
                    info = _quant_payload(f.get_tensor(qkey))
                    quant_format = info.get("format")
                    scale_key = comfy_scale_key_for_weight(key)
                    if quant_format == "nvfp4":
                        out_f, in_f = int(tensor.shape[0]), int(tensor.shape[1]) * 2
                        tensor = dequantize_nvfp4_tensor(tensor, f.get_tensor(scale_key), f.get_tensor(comfy_scale2_key_for_weight(key)), out_f, in_f, torch.bfloat16, torch.device("cpu"))
                    elif quant_format in {"int8_tensorwise", "float8_e4m3fn", "float8_e5m2"}:
                        if scale_key not in keys:
                            raise ValueError(f"Missing quantization scale for {key}")
                        tensor = (tensor.float() * f.get_tensor(scale_key).float()).to(torch.bfloat16)
                    else:
                        raise ValueError(f"Projected repair does not support {quant_format!r} at {key}")
                elif torch.is_floating_point(tensor):
                    tensor = tensor.to(torch.bfloat16)
                records.append((key, tensor.cpu().contiguous(), tensor.dtype))
        write_streaming_safetensors(view_path, records, metadata)
        del records
        gc.collect()
    config.ANIMA_PROJECTED_QUANT_SOURCE = str(quant_path)
    return str(view_path)
def load_anima_pipe(config, device):
    ensure_training_utils_available()

    original_dit_path = config.DIT_PATH
    target_format = _anima_qat_target_format(config)
    if target_format == "bfloat16":
        compute_view_path = str(original_dit_path)
        config.ANIMA_PROJECTED_QUANT_SOURCE = str(original_dit_path)
        print("INFO: BF16 repair-control mode: loading the base DiT directly.")
    else:
        compute_view_path = _prepare_quantized_training_view(config, original_dit_path)


    model_configs = [
        ModelConfig(path=compute_view_path),
        ModelConfig(path=config.TEXT_ENCODER_PATH),
        ModelConfig(path=config.VAE_PATH),
    ]

    tokenizer_config = make_anima_tokenizer_config(config.TOKENIZER_PATH, "Qwen")
    tokenizer_t5xxl_config = make_anima_tokenizer_config(config.TOKENIZER_T5XXL_PATH, "T5XXL")

    loaded_template_path = compute_view_path
    try:
        pipe = AnimaImagePipeline.from_pretrained(
            torch_dtype=config.compute_dtype,
            device=device,
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
            tokenizer_t5xxl_config=tokenizer_t5xxl_config,
        )
    except ValueError as e:
        message = str(e)
        if "Cannot detect the model type" not in message:
            raise
        raise RuntimeError(
            "The selected Anima DiT file could not be loaded by training_utils. "
            "The file is likely missing required architecture metadata or was saved in an unsupported/corrupted format. "
            "Repair or re-export the DiT checkpoint with proper Anima metadata, then select the repaired file."
        ) from e
    else:
        loaded_prefix = detect_anima_dit_key_prefix(original_dit_path)

    required_components = {
        "dit": original_dit_path,
        "vae": config.VAE_PATH,
        "text_encoder": config.TEXT_ENCODER_PATH,
    }
    missing_components = [name for name in required_components if getattr(pipe, name, None) is None]
    if missing_components:
        details = ", ".join(f"{name} from {required_components[name]}" for name in missing_components)
        raise RuntimeError(
            "training_utils loaded the Anima pipeline, but one or more required components are missing: "
            f"{details}. Check that each selected file is the correct Anima component type."
        )

    for attr in required_components:
        getattr(pipe, attr).cpu()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    config.ANIMA_LOADED_DIT_TEMPLATE_PATH = loaded_template_path
    if str(getattr(config, "ANIMA_DIT_SAVE_PREFIX", "auto")).lower() == "auto":
        config.ANIMA_DIT_SAVE_PREFIX = loaded_prefix
    return pipe


def apply_anima_t5_token_dropout(t5xxl_ids, captions, config, pad_id=0):
    if config is None or not getattr(config, "T5_TOKEN_DROPOUT_ENABLED", False):
        return t5xxl_ids

    chance = min(max(float(getattr(config, "T5_TOKEN_DROPOUT_CHANCE", 0.0) or 0.0), 0.0), 1.0)
    min_rate = min(max(float(getattr(config, "T5_TOKEN_DROPOUT_MIN", 0.0) or 0.0), 0.0), 1.0)
    max_rate = min(max(float(getattr(config, "T5_TOKEN_DROPOUT_MAX", 0.0) or 0.0), 0.0), 1.0)
    if max_rate < min_rate:
        min_rate, max_rate = max_rate, min_rate
    if chance <= 0.0 or max_rate <= 0.0:
        return t5xxl_ids

    out = t5xxl_ids.clone()
    for batch_index, caption in enumerate(captions):
        ids = out[batch_index]
        candidates = torch.ones_like(ids, dtype=torch.bool)
        candidates &= ids.ne(pad_id)
        if not candidates.any():
            continue
        seed_base = int(getattr(config, "SEED", 42) or 42)
        digest = hashlib.sha256(f"{seed_base}:t5:{caption}".encode("utf-8", errors="ignore")).digest()
        seed = int.from_bytes(digest[:8], "little") % (2 ** 63)
        generator = torch.Generator(device=ids.device)
        generator.manual_seed(seed)
        if torch.rand((), device=ids.device, generator=generator).item() >= chance:
            continue
        rate = min_rate + (max_rate - min_rate) * torch.rand((), device=ids.device, generator=generator).item()
        candidate_indices = torch.nonzero(candidates, as_tuple=False).flatten()
        drop_count = int(round(candidate_indices.numel() * rate))
        if drop_count <= 0:
            continue
        perm = torch.randperm(candidate_indices.numel(), device=ids.device, generator=generator)
        ids[candidate_indices[perm[:drop_count]]] = pad_id
    return out


@torch.no_grad()
def encode_prompt_anima(pipe, caption, device, config=None):
    if isinstance(caption, str):
        caption = [caption]

    text_inputs = pipe.tokenizer(
        caption,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids.to(device)
    prompt_masks = text_inputs.attention_mask.to(device).bool()
    prompt_embeds = pipe.text_encoder(
        input_ids=text_input_ids,
        attention_mask=prompt_masks,
        output_hidden_states=True,
    ).hidden_states[-1]

    t5xxl_text_inputs = pipe.tokenizer_t5xxl(
        caption,
        max_length=512,
        truncation=True,
        return_tensors="pt",
    )
    t5xxl_ids = t5xxl_text_inputs.input_ids.to(device)
    t5xxl_ids = apply_anima_t5_token_dropout(t5xxl_ids, caption, config, pad_id=getattr(pipe.tokenizer_t5xxl, "pad_token_id", 0) or 0)
    return prompt_embeds.to(pipe.torch_dtype), t5xxl_ids


@torch.no_grad()
def encode_image_anima(pipe, image, device, tiled=True, tile_size=(96, 96), tile_stride=(72, 72), dtype=None):
    dtype = dtype or pipe.torch_dtype
    image_tensor = pipe.preprocess_image(image).to(device="cpu", dtype=dtype)
    latents = pipe.vae.encode(
        image_tensor.unsqueeze(2),
        device=device,
        tiled=tiled,
        tile_size=tuple(tile_size),
        tile_stride=tuple(tile_stride),
    ).squeeze(2)
    return latents


def save_anima_null_conditioning_cache(cache_dir, null_prompt_emb, null_t5xxl_ids, text_cache_dtype):
    torch.save(
        {
            "prompt_emb": null_prompt_emb[0].to(dtype=text_cache_dtype).cpu(),
            "t5xxl_ids": null_t5xxl_ids[0].to(dtype=torch.long).cpu(),
        },
        cache_dir / "null_embeds.pt",
    )


def ensure_anima_null_conditioning_cache(config, pipe, device):
    if not null_conditioning_cache_needed(config):
        return

    cache_name = anima_cache_folder_name(config)
    missing_cache_dirs = []
    for root in anima_dataset_roots(config):
        cache_dir = root / cache_name
        if cache_index_exists(cache_dir) and not (cache_dir / "null_embeds.pt").exists():
            missing_cache_dirs.append(cache_dir)

    if not missing_cache_dirs:
        return

    print(f"INFO: Creating Anima null conditioning cache for {len(missing_cache_dirs)} dataset(s).")
    if hasattr(pipe, "dit"):
        pipe.dit.cpu()
    pipe.vae.cpu()
    pipe.text_encoder.to(device=device, dtype=config.compute_dtype)
    pipe.text_encoder.eval()

    with torch.no_grad():
        null_prompt_emb, null_t5xxl_ids = encode_prompt_anima(pipe, "", device)
    text_cache_dtype = anima_text_cache_float_dtype(config)
    for cache_dir in missing_cache_dirs:
        save_anima_null_conditioning_cache(cache_dir, null_prompt_emb, null_t5xxl_ids, text_cache_dtype)

    pipe.text_encoder.cpu()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def precompute_and_cache_anima(config, pipe, device):
    roots_to_rebuild = anima_roots_needing_cache_rebuild(config)
    if not roots_to_rebuild:
        ensure_anima_null_conditioning_cache(config, pipe, device)
        print("\n" + "=" * 60 + "\nINFO: Anima DiT datasets already cached.\n" + "=" * 60 + "\n")
        return

    cache_name = anima_cache_folder_name(config)
    text_cache_dtype = anima_text_cache_float_dtype(config)
    vae_cache_dtype = anima_vae_cache_float_dtype(config)
    expected_options = get_anima_cache_options(config)
    caption_mode = caption_source_type(config)
    json_caption_mode = json_caption_mode_enabled(config)
    active_caption_types = anima_caption_types_for_cache(config, json_caption_mode)
    print(
        "INFO: Anima cache precision: "
        f"text={anima_text_cache_float_dtype_name(config)}, "
        f"vae={anima_vae_cache_float_dtype_name(config)}."
    )
    multi_bucket_extra = (
        max(0, int(getattr(config, "MULTI_BUCKET_EXTRA_BUCKETS", 0) or 0))
        if getattr(config, "MULTI_BUCKET_ENABLED", False)
        else 0
    )
    if multi_bucket_extra > 0:
        print(f"INFO: Multi-bucket cache enabled: up to {multi_bucket_extra} extra bucket(s) per image.")

    for root in roots_to_rebuild:
        if not root.exists():
            print(f"WARNING: Dataset folder does not exist: {root}")
            continue

        cache_dir = root / cache_name
        cache_dir.mkdir(parents=True, exist_ok=True)
        force_recaching = bool(getattr(config, "REBUILD_CACHE", False))
        existing_index = None
        if cache_index_exists(cache_dir) and not force_recaching:
            try:
                existing_index = load_cache_index(cache_dir)
                if not anima_image_layout_options_match(existing_index.get("cache_options"), expected_options):
                    print(
                        f"INFO: Anima cache options changed for {root.name}; "
                        "reusing compatible cached files and filling only missing variants."
                    )
            except Exception:
                print(f"INFO: Anima cache index for {root.name} is unreadable; rebuilding index from cache files.")

        image_paths = collect_anima_image_paths(root)
        current_cache_stems = {anima_cache_stem_for_image(root, p) for p in image_paths}
        if not image_paths:
            stale_files = list(cache_dir.glob("*.pt"))
            if stale_files:
                print(f"INFO: Removing {len(stale_files)} stale Anima cache item(s) because {root.name} has no images.")
            for f in stale_files:
                remove_anima_cache_file(f)
            save_cache_index(cache_dir, {"version": 6, "cache_options": expected_options, "files": []})
            print(f"WARNING: No images found in {root}")
            continue

        if force_recaching:
            for f in cache_dir.glob("*.pt"):
                remove_anima_cache_file(f)
        else:
            stale_files = [
                f for f in cache_dir.glob("*.pt")
                if f.name not in {"null_embeds.pt", "dataset_index.pt"} and anima_cache_base_stem(f) not in current_cache_stems
            ]
            if stale_files:
                print(f"INFO: Removing {len(stale_files)} stale Anima cache item(s) for deleted images in {root.name}.")
            for f in stale_files:
                remove_anima_cache_file(f)

        max_bucket_resolution = get_max_bucket_resolution_for_config(config)
        max_bucket_area = max_bucket_resolution * max_bucket_resolution
        existing_items_by_relative_path = defaultdict(list)
        deep_cache_validate = bool(getattr(config, "ANIMA_DEEP_CACHE_VALIDATE", False))
        can_reuse_index_metadata = (
            existing_index is not None
            and not force_recaching
            and anima_image_layout_options_match(existing_index.get("cache_options"), expected_options)
        )
        if can_reuse_index_metadata and deep_cache_validate:
            can_reuse_index_metadata = cache_payload_options_match_for_index(
                existing_index,
                expected_options,
                anima_cache_paths_for_index_item,
                anima_text_cache_compatible_options,
                anima_lat_cache_compatible_options,
            )
        if can_reuse_index_metadata:
            for item in existing_index.get("files", []):
                relative_path = item.get("relative_path")
                if relative_path:
                    existing_items_by_relative_path[relative_path].append(item)

        reused_index_items = []
        image_paths_to_validate = []
        for image_path in image_paths:
            relative_path = str(image_path.relative_to(root))
            existing_items = existing_items_by_relative_path.get(relative_path, [])
            if existing_items:
                signatures_match = cached_file_signatures_match(
                    existing_items[0],
                    image_path,
                    caption_mode,
                )
                existing_paths = [
                    path
                    for item in existing_items
                    for path in anima_cache_paths_for_index_item(item)
                ]
                if signatures_match and existing_paths and all(Path(path).exists() for path in existing_paths):
                    reused_index_items.extend(existing_items)
                    continue
            image_paths_to_validate.append(image_path)

        if reused_index_items:
            print(
                f"INFO: Reusing Anima cache metadata for {len(reused_index_items)} indexed bucket item(s) "
                f"from unchanged images in {root.name}."
            )

        if image_paths_to_validate:
            print(
                f"INFO: Validating Anima DiT images and assigning buckets for {root.name} "
                f"({len(image_paths_to_validate)}/{len(image_paths)} image(s))..."
            )
            with Pool(processes=min(cpu_count(), 8)) as pool:
                results = list(tqdm(
                    pool.imap(
                        validate_and_assign_resolution,
                        [(p, max_bucket_area, 64, config.SHOULD_UPSCALE, caption_mode) for p in image_paths_to_validate],
                    ),
                    total=len(image_paths_to_validate),
                ))
        else:
            results = []

        expanded_results = []
        for m in (r for r in results if r):
            buckets = get_multi_bucket_resolutions(
                m["original_size"][0],
                m["original_size"][1],
                max_bucket_area,
                config.SHOULD_UPSCALE,
                multi_bucket_extra,
            )
            for variant_index, (target_w, target_h) in enumerate(buckets):
                variant_meta = make_bucket_variant_metadata(m, target_w, target_h, variant_index)
                variant_meta["root"] = root
                expanded_results.append(variant_meta)

        index_data = []
        text_jobs = []
        lat_jobs = []
        reused_text_count = 0
        reused_lat_count = 0
        expected_text_count = 0
        expected_cache_files = set()
        for item in reused_index_items:
            for cache_path in anima_cache_paths_for_index_item(item):
                expected_cache_files.add(Path(cache_path).resolve())
            index_data.append(item)

        for m in expanded_results:
            caption_variants = m.get("caption_variants") or {"txt": m["caption"]}
            item_caption_types = tuple(key for key in active_caption_types if key in caption_variants)
            if not item_caption_types:
                fallback_caption_type = (
                    CAPTION_JSON_PRIMARY_TYPE
                    if CAPTION_JSON_PRIMARY_TYPE in caption_variants
                    else next(iter(caption_variants))
                )
                item_caption_types = (fallback_caption_type,)
            text_paths, lat_path = anima_expected_cache_paths(root, cache_dir, m, item_caption_types, json_caption_mode)
            expected_cache_files.add(lat_path.resolve())
            variant_cache_paths = {}
            expected_text_count += len(item_caption_types)
            for caption_type in item_caption_types:
                te_path = text_paths[caption_type]
                expected_cache_files.add(te_path.resolve())
                variant_cache_paths[caption_type] = str(te_path)
                caption = caption_variants[caption_type]
                if te_path.exists() and anima_text_cache_valid(te_path, m, caption_type, caption, text_cache_dtype, expected_options):
                    reused_text_count += 1
                else:
                    text_jobs.append((m, caption_type, caption, te_path))

            if lat_path.exists() and anima_lat_cache_valid(lat_path, m, vae_cache_dtype, expected_options):
                reused_lat_count += 1
            else:
                lat_jobs.append((m, lat_path))

            primary_te_path = variant_cache_paths.get(CAPTION_JSON_PRIMARY_TYPE) or next(iter(variant_cache_paths.values()))
            item = {
                "te_path": primary_te_path,
                "lat_path": str(lat_path),
                "relative_path": str(m["ip"].relative_to(root)),
                "image_file_signature": image_file_signature(m["ip"]),
                "caption_file_signature": caption_file_signature_for_image(m["ip"], caption_mode),
                "target_size": tuple(m["target_resolution"]),
                "original_size": m["original_size"],
                "scaled_size": m["scaled_size"],
                "crop_coords": m.get("crop_coords", (0, 0)),
                "bucket_variant_index": m.get("bucket_variant_index", 0),
                "caption_signature": m.get("caption_signature"),
            }
            if json_caption_mode:
                item["caption_variants"] = anima_caption_variant_index(variant_cache_paths)
            index_data.append(item)

        obsolete_files = [
            f for f in cache_dir.glob("*.pt")
            if f.name not in {"null_embeds.pt", "dataset_index.pt"}
            and anima_cache_base_stem(f) in current_cache_stems
            and f.resolve() not in expected_cache_files
        ]
        if obsolete_files:
            print(f"INFO: Removing {len(obsolete_files)} obsolete Anima cache variant(s) for {root.name}.")
        for f in obsolete_files:
            remove_anima_cache_file(f)

        if not text_jobs and not lat_jobs:
            save_cache_index(cache_dir, {"version": 6, "cache_options": expected_options, "files": index_data})
            print(f"INFO: Anima DiT cache current for {root.name}: {len(index_data)} item(s).")
            continue

        print(
            f"INFO: Anima cache reuse for {root.name}: "
            f"{reused_text_count}/{expected_text_count} text item(s), "
            f"{reused_lat_count}/{len(expanded_results)} latent item(s)."
        )

        if hasattr(pipe, "dit"):
            pipe.dit.cpu()
        pipe.vae.cpu()
        pipe.text_encoder.cpu()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"INFO: Anima cache phase 1/2: text encoder for {root.name}")
        if text_jobs or null_conditioning_cache_needed(config):
            pipe.text_encoder.to(device=device, dtype=config.compute_dtype)
            pipe.text_encoder.eval()
            pipe.vae.cpu()

        if null_conditioning_cache_needed(config) and (text_jobs or not (cache_dir / "null_embeds.pt").exists()):
            with torch.no_grad():
                null_prompt_emb, null_t5xxl_ids = encode_prompt_anima(pipe, "", device)
        else:
            null_prompt_emb, null_t5xxl_ids = None, None

        if text_jobs:
            with torch.no_grad():
                with tqdm(total=len(text_jobs), desc=f"Caching Anima text {root.name}", unit="item") as pbar:
                    for m, caption_type, caption, te_path in text_jobs:
                        prompt_emb, t5xxl_ids = encode_prompt_anima(pipe, caption, device)
                        torch.save({
                            "prompt_emb": prompt_emb[0].to(dtype=text_cache_dtype).cpu(),
                            "t5xxl_ids": t5xxl_ids[0].to(dtype=torch.long).cpu(),
                            "caption_type": caption_type,
                            "caption": caption,
                            "caption_signature": m.get("caption_signature"),
                            "image_path": str(m["ip"]),
                            "relative_path": str(m["ip"].relative_to(root)),
                            "image_file_signature": image_file_signature(m["ip"]),
                            "caption_file_signature": caption_file_signature_for_image(m["ip"], caption_mode),
                            "original_size": m["original_size"],
                            "scaled_size": m["scaled_size"],
                            "target_size": tuple(m["target_resolution"]),
                            "crop_coords": m.get("crop_coords", (0, 0)),
                            "bucket_variant_index": m.get("bucket_variant_index", 0),
                            "cache_options": expected_options,
                        }, te_path)
                        pbar.update(1)

        if null_prompt_emb is not None and null_t5xxl_ids is not None:
            save_anima_null_conditioning_cache(cache_dir, null_prompt_emb, null_t5xxl_ids, text_cache_dtype)

        pipe.text_encoder.cpu()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        tile_size = tuple(getattr(config, "VAE_CACHING_TILE_SIZE", [96, 96]))
        tile_stride = tuple(getattr(config, "VAE_CACHING_TILE_STRIDE", [72, 72]))
        if lat_jobs:
            print(
                "INFO: Anima cache phase 2/2: VAE "
                f"(tiled={getattr(config, 'VAE_CACHING_TILED', True)}, "
                f"tile_size={tile_size}, tile_stride={tile_stride})"
            )
            pipe.vae.to(device=device, dtype=vae_cache_dtype)
            pipe.vae.eval()

            with torch.inference_mode():
                with tqdm(total=len(lat_jobs), desc=f"Caching Anima VAE {root.name}", unit="item") as pbar:
                    for m, lat_path in lat_jobs:
                        image_path = Path(m["ip"])
                        w, h = m["target_resolution"]
                        try:
                            with Image.open(image_path) as img:
                                image = smart_resize(fix_alpha_channel(img), w, h)
                            latents = encode_image_anima(
                                pipe,
                                image,
                                device,
                                tiled=bool(getattr(config, "VAE_CACHING_TILED", True)),
                                tile_size=tile_size,
                                tile_stride=tile_stride,
                                dtype=vae_cache_dtype,
                            )
                            cached_latents = latents[0].to(dtype=vae_cache_dtype).cpu()
                            lineart_mask = (
                                generate_lineart_loss_map(image, cached_latents.shape[-2], cached_latents.shape[-1])
                                if ANIMA_REPAIR_LINEART_LOSS_ENABLED
                                else None
                            )
                            torch.save({
                                "latents": cached_latents,
                                "lineart_mask": lineart_mask,
                            "image_path": str(image_path),
                            "relative_path": str(m["ip"].relative_to(root)),
                            "image_file_signature": image_file_signature(m["ip"]),
                            "caption_file_signature": caption_file_signature_for_image(m["ip"], caption_mode),
                            "original_size": m["original_size"],
                                "scaled_size": m["scaled_size"],
                                "target_size": tuple(m["target_resolution"]),
                                "crop_coords": m.get("crop_coords", (0, 0)),
                                "bucket_variant_index": m.get("bucket_variant_index", 0),
                                "caption_signature": m.get("caption_signature"),
                                "cache_options": expected_options,
                            }, lat_path)
                        except Exception as e:
                            print(f"[SKIP ANIMA VAE] {image_path.name}: {e}")
                            remove_anima_cache_file(lat_path)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        pbar.update(1)
        else:
            print(f"INFO: Anima cache phase 2/2: VAE cache already current for {root.name}.")

        pipe.vae.cpu()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        valid_index_data = [
            item for item in index_data
            if all(Path(path).exists() for path in anima_cache_paths_for_index_item(item))
        ]
        save_cache_index(cache_dir, {"version": 6, "cache_options": expected_options, "files": valid_index_data})
        print(f"INFO: Cached {len(valid_index_data)} Anima DiT items to {cache_dir}")

    ensure_anima_null_conditioning_cache(config, pipe, device)


class AnimaCachedDataset(Dataset):
    SAMPLE_INDEX_BITS = 32
    SAMPLE_INDEX_MASK = (1 << SAMPLE_INDEX_BITS) - 1

    def __init__(self, config):
        self.items = []
        self.bucket_keys = []
        self.seed = config.SEED if config.SEED else 42
        self.json_caption_mode = json_caption_mode_enabled(config)
        self.caption_weights = get_json_caption_weights(config)
        cache_name = anima_cache_folder_name(config)
        self.null_prompt_emb = None
        self.null_t5xxl_ids = None
        self.cond_scale_min, self.cond_scale_max = get_text_conditioning_scale_range(config)
        self.cond_scale_enabled = self.cond_scale_min < 1.0 or self.cond_scale_max > 1.0
        null_dropout_enabled = bool(getattr(config, "UNCONDITIONAL_DROPOUT", False))
        self.qwen_null_dropout_prob = (
            min(max(float(getattr(config, "QWEN_NULL_DROPOUT_CHANCE", 0.0) or 0.0), 0.0), 1.0)
            if null_dropout_enabled
            else 0.0
        )
        self.t5_null_dropout_prob = (
            min(max(float(getattr(config, "T5_NULL_DROPOUT_CHANCE", 0.0) or 0.0), 0.0), 1.0)
            if null_dropout_enabled
            else 0.0
        )
        self.t5_token_dropout_enabled = bool(getattr(config, "T5_TOKEN_DROPOUT_ENABLED", False))
        self.t5_token_dropout_chance = min(max(float(getattr(config, "T5_TOKEN_DROPOUT_CHANCE", 0.0) or 0.0), 0.0), 1.0)
        self.t5_token_dropout_min = min(max(float(getattr(config, "T5_TOKEN_DROPOUT_MIN", 0.0) or 0.0), 0.0), 1.0)
        self.t5_token_dropout_max = min(max(float(getattr(config, "T5_TOKEN_DROPOUT_MAX", 0.0) or 0.0), 0.0), 1.0)
        if self.t5_token_dropout_max < self.t5_token_dropout_min:
            self.t5_token_dropout_min, self.t5_token_dropout_max = self.t5_token_dropout_max, self.t5_token_dropout_min

        for ds in getattr(config, "INSTANCE_DATASETS", []):
            root = Path(ds["path"])
            cache_dir = root / cache_name
            if not cache_index_exists(cache_dir):
                print(f"WARNING: Anima DiT index missing at {cache_dir}.")
                continue
            index_data = load_cache_index(cache_dir)
            repeats = int(ds.get("repeats", 1))
            for _ in range(repeats):
                for item in index_data["files"]:
                    self.items.append(item)
                    self.bucket_keys.append(tuple(item["target_size"]))

        if not self.items:
            raise ValueError("No cached Anima DiT files found.")

        combined = list(zip(self.items, self.bucket_keys))
        random.Random(self.seed).shuffle(combined)
        self.items, self.bucket_keys = zip(*combined)
        self.items = list(self.items)
        self.bucket_keys = list(self.bucket_keys)
        if self.qwen_null_dropout_prob > 0 or self.t5_null_dropout_prob > 0 or self.cond_scale_enabled:
            try:
                null_data = torch.load(
                    Path(config.INSTANCE_DATASETS[0]["path"]) / cache_name / "null_embeds.pt",
                    map_location="cpu",
                    weights_only=True,
                )
                self.null_prompt_emb = null_data["prompt_emb"].squeeze(0) if null_data["prompt_emb"].dim() == 3 else null_data["prompt_emb"]
                null_t5xxl_ids = null_data["t5xxl_ids"]
                if null_t5xxl_ids.dim() == 0:
                    null_t5xxl_ids = null_t5xxl_ids.view(1)
                elif null_t5xxl_ids.dim() > 1 and null_t5xxl_ids.shape[0] == 1:
                    null_t5xxl_ids = null_t5xxl_ids.squeeze(0)
                self.null_t5xxl_ids = null_t5xxl_ids.to(dtype=torch.long)
            except Exception:
                self.qwen_null_dropout_prob = 0.0
                self.t5_null_dropout_prob = 0.0
                self.cond_scale_enabled = False

    def __len__(self):
        return len(self.items)

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
        payload = f"{self.seed}:anima-sample:{int(sample_index)}:{int(dataset_index)}".encode("utf-8")
        digest = hashlib.sha256(payload).digest()
        return random.Random(int.from_bytes(digest[:8], "little"))

    def _align_null_prompt_emb(self, prompt_emb):
        null_prompt_emb = self.null_prompt_emb
        if null_prompt_emb is None or prompt_emb.shape == null_prompt_emb.shape:
            return prompt_emb, null_prompt_emb
        if prompt_emb.dim() != 2 or null_prompt_emb.dim() != 2 or prompt_emb.shape[1] != null_prompt_emb.shape[1]:
            return prompt_emb, null_prompt_emb

        prompt_len = prompt_emb.shape[0]
        null_len = null_prompt_emb.shape[0]
        if prompt_len < null_len:
            prompt_pad = null_prompt_emb[prompt_len:null_len].to(dtype=prompt_emb.dtype)
            prompt_emb = torch.cat([prompt_emb, prompt_pad], dim=0)
        elif prompt_len > null_len:
            pad = null_prompt_emb[-1:].expand(prompt_len - null_len, -1)
            null_prompt_emb = torch.cat([null_prompt_emb, pad], dim=0)
        return prompt_emb, null_prompt_emb.to(dtype=prompt_emb.dtype)

    def _apply_t5_token_dropout(self, ids, rng):
        if (
            not self.t5_token_dropout_enabled
            or self.t5_token_dropout_chance <= 0.0
            or self.t5_token_dropout_max <= 0.0
            or rng.random() >= self.t5_token_dropout_chance
        ):
            return ids

        candidates = torch.nonzero(ids.ne(0), as_tuple=False).flatten().tolist()
        if not candidates:
            return ids

        rate = rng.uniform(self.t5_token_dropout_min, self.t5_token_dropout_max)
        drop_count = int(round(len(candidates) * rate))
        if drop_count <= 0:
            return ids

        out = ids.clone()
        for token_index in rng.sample(candidates, min(drop_count, len(candidates))):
            out[token_index] = 0
        return out

    def __getitem__(self, i):
        try:
            dataset_index, sample_index = self.unpack_sample_index(i)
            rng = self._rng_for_sample(dataset_index, sample_index)
            item = self.items[dataset_index]
            te_path = selected_caption_variant_path(
                item,
                rng,
                self.caption_weights,
                enabled=self.json_caption_mode,
            )
            lat_path = item["lat_path"]
            data_te = torch.load(te_path, map_location="cpu", weights_only=True)
            data_lat = torch.load(lat_path, map_location="cpu", weights_only=True)
            latents = data_lat["latents"]
            if torch.isnan(latents).any() or torch.isinf(latents).any():
                return None
            item_data = {
                "latents": latents,
                "prompt_emb": data_te["prompt_emb"],
                "t5xxl_ids": data_te["t5xxl_ids"],
                "target_size": item["target_size"],
                "latent_path": lat_path,
                "image_key": item.get("relative_path", lat_path),
                "lineart_mask": data_lat.get("lineart_mask"),
            }

            qwen_dropped = False
            if self.qwen_null_dropout_prob > 0 and rng.random() < self.qwen_null_dropout_prob:
                _, null_prompt_emb = self._align_null_prompt_emb(item_data["prompt_emb"])
                if null_prompt_emb is not None:
                    item_data["prompt_emb"] = null_prompt_emb
                    qwen_dropped = True
            if self.t5_null_dropout_prob > 0 and rng.random() < self.t5_null_dropout_prob:
                if self.null_t5xxl_ids is not None:
                    item_data["t5xxl_ids"] = self.null_t5xxl_ids
            else:
                item_data["t5xxl_ids"] = self._apply_t5_token_dropout(item_data["t5xxl_ids"], rng)
            if not qwen_dropped and self.cond_scale_enabled:
                scale = rng.uniform(self.cond_scale_min, self.cond_scale_max)
                prompt_emb, null_prompt_emb = self._align_null_prompt_emb(item_data["prompt_emb"])
                if null_prompt_emb is not None:
                    item_data["prompt_emb"] = null_prompt_emb + (prompt_emb - null_prompt_emb) * scale

            return item_data
        except Exception as e:
            print(f"[ANIMA DATASET] Failed to load item {i}: {e}")
            return None


def anima_collate_fn(batch):
    batch = list(filter(None, batch))
    if not batch:
        return {}

    for item in batch:
        if item["t5xxl_ids"].dim() == 0:
            item["t5xxl_ids"] = item["t5xxl_ids"].view(1)
    max_t5_len = max(item["t5xxl_ids"].shape[0] for item in batch)
    t5_ids = []
    for item in batch:
        ids = item["t5xxl_ids"]
        if ids.shape[0] < max_t5_len:
            ids = torch.cat([ids, torch.zeros(max_t5_len - ids.shape[0], dtype=ids.dtype)], dim=0)
        t5_ids.append(ids)

    return {
        "latents": torch.stack([item["latents"] for item in batch]),
        "prompt_emb": torch.stack([item["prompt_emb"] for item in batch]),
        "t5xxl_ids": torch.stack(t5_ids),
        "target_size": [item["target_size"] for item in batch],
        "latent_path": [item["latent_path"] for item in batch],
        "image_key": [item["image_key"] for item in batch],
        "lineart_mask": (
            torch.stack([item["lineart_mask"] for item in batch])
            if all(item.get("lineart_mask") is not None for item in batch)
            else None
        ),
    }


def print_anima_dataset_resolution_sample(dataset, sample_count=5):
    sample_count = min(sample_count, len(dataset.items))
    if sample_count <= 0:
        return

    print(f"INFO: Anima dataset resolution sample ({sample_count} cached item{'s' if sample_count != 1 else ''}):")
    for item in dataset.items[:sample_count]:
        orig_w, orig_h = item["original_size"]
        targ_w, targ_h = item["target_size"]
        orig_ar = orig_w / orig_h if orig_h else 1.0
        targ_ar = targ_w / targ_h if targ_h else 1.0
        ar_error_pct = (abs(orig_ar - targ_ar) / orig_ar * 100) if orig_ar else 0.0
        target_area = targ_w * targ_h
        stem = Path(item["lat_path"]).stem.replace("_lat", "")
        variant_label = f", variant {item.get('bucket_variant_index', 0)}" if item.get("bucket_variant_index", 0) else ""
        print(
            f"INFO:   {stem}: original {orig_w}x{orig_h} (AR {orig_ar:.4f}) -> "
            f"target {targ_w}x{targ_h} ({target_area:,} px, AR {targ_ar:.4f})"
            f"{variant_label}, AR diff {ar_error_pct:.2f}%, cropped not stretched"
        )


def pack_anima_sample_schedule(image_schedule, batch_size):
    packed_schedule = []
    batch_size = max(1, int(batch_size or 1))
    for batch_index, batch in enumerate(image_schedule):
        packed_batch = []
        for local_index, dataset_index in enumerate(batch):
            sample_index = batch_index * batch_size + local_index
            packed_batch.append(AnimaCachedDataset.pack_sample_index(dataset_index, sample_index))
        packed_schedule.append(packed_batch)
    return packed_schedule


class AnimaTimestepSampler:
    def __init__(self, config, total_timestep_count):
        self.config = config
        self.total_timestep_count = total_timestep_count
        self.total_tickets_needed = config.MAX_TRAIN_STEPS * config.BATCH_SIZE
        self.seed = config.SEED if config.SEED else 42
        self.bin_ranges = []
        self.ticket_pool = self._build_ticket_pool(getattr(config, "TIMESTEP_ALLOCATION", None))
        self.pool_index = 0

    def _build_ticket_pool(self, allocation):
        pool, self.bin_ranges = build_timestep_ticket_pool(
            allocation,
            self.total_tickets_needed,
            self.total_timestep_count,
            self.seed,
            bool(getattr(self.config, "TIMESTEP_STRATIFIED_SAMPLING", False)),
        )
        return pool

    def set_current_step(self, micro_step):
        self.pool_index = (micro_step * self.config.BATCH_SIZE) % len(self.ticket_pool)

    def _sample_from_pool(self):
        if self.pool_index >= len(self.ticket_pool):
            self.pool_index = 0
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
        return torch.tensor(indices, dtype=torch.long), indices[0]


def freeze_dit_layers(dit, exclude_patterns):
    total_params = frozen_params = trainable_params = 0
    for name, param in dit.named_parameters():
        total_params += param.numel()
        should_freeze = any(fnmatch.fnmatch(name, pattern if "*" in pattern else f"*{pattern}*") for pattern in exclude_patterns)
        param.requires_grad_(not should_freeze)
        if should_freeze:
            frozen_params += param.numel()
        else:
            trainable_params += param.numel()

    print("=" * 56)
    print("INFO: DiT Parameter Statistics")
    print(f"- Total Parameters:     {total_params:,}")
    print(f"- Frozen Parameters:    {frozen_params:,}")
    print(f"- Trainable Parameters: {trainable_params:,}")
    print(f"- Percentage Frozen:    {(frozen_params / max(total_params, 1)) * 100:.2f}%")
    print("=" * 56)


def _anima_qat_target_format(config):
    aliases = {
        "bf16": "bfloat16", "bfp16": "bfloat16", "bfloat16": "bfloat16",
        "4": "nvfp4", "fp4": "nvfp4", "nvfp4": "nvfp4",
        "8": "int8_tensorwise", "int8": "int8_tensorwise", "int8_tensorwise": "int8_tensorwise",
        "e4m3": "float8_e4m3fn", "fp8": "float8_e4m3fn", "float8_e4m3fn": "float8_e4m3fn",
        "e5m2": "float8_e5m2", "float8_e5m2": "float8_e5m2",
    }
    requested = str(ANIMA_QAT_TARGET_FORMAT).strip().lower()
    if requested not in aliases:
        raise ValueError(
            f"Unsupported ANIMA_QAT_TARGET_FORMAT={requested!r}; use bf16, nvfp4, int8, e4m3, or e5m2."
        )
    return aliases[requested]


def _anima_qat_scale_multiplier(config):
    value = float(ANIMA_QAT_NVFP4_SCALE_MULTIPLIER)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("ANIMA_QAT_NVFP4_SCALE_MULTIPLIER must be positive and finite.")
    return value


class AnimaProjectedQuantController:
    """Owns projected quantized state and floating error-feedback residuals."""
    def __init__(self, dit, config):
        from scripts.convert_anima_to_quants import comfy_quant_key_for_weight, comfy_scale2_key_for_weight, comfy_scale_key_for_weight
        self.format = _anima_qat_target_format(config)
        self.scale_multiplier = _anima_qat_scale_multiplier(config)
        self.source_path = Path(getattr(config, "ANIMA_PROJECTED_QUANT_SOURCE", config.DIT_PATH))
        module_map = {name: module for name, module in dit.named_modules() if isinstance(module, torch.nn.Linear)}
        prefix = detect_anima_dit_key_prefix(self.source_path)
        self.modules, self.weight_keys, self.packed, self.residuals = {}, {}, {}, {}
        with safe_open(str(self.source_path), framework="pt", device="cpu") as f:
            keys = set(f.keys())
            for qkey in sorted(k for k in keys if k.endswith(".comfy_quant")):
                source_format = _quant_payload(f.get_tensor(qkey)).get("format")
                if source_format not in {"nvfp4", "int8_tensorwise", "float8_e4m3fn", "float8_e5m2"}:
                    continue
                weight_key = qkey[:-len(".comfy_quant")] + ".weight"
                candidate = weight_key[:-len(".weight")]
                if prefix and candidate.startswith(prefix):
                    candidate = candidate[len(prefix):]
                name = candidate if candidate in module_map else None
                if name is None:
                    matches = [n for n in module_map if n.endswith("." + candidate)]
                    name = matches[0] if len(matches) == 1 else None
                if name is None or weight_key not in keys:
                    continue
                scale_key = comfy_scale_key_for_weight(weight_key)
                scale2_key = comfy_scale2_key_for_weight(weight_key)
                if scale_key not in keys or (source_format == "nvfp4" and scale2_key not in keys):
                    raise ValueError(f"Missing {source_format} scales for {weight_key}")
                module = module_map[name]
                self.modules[name] = module
                self.weight_keys[name] = weight_key
                packed = {
                    "weight": f.get_tensor(weight_key).cpu().contiguous(),
                    "weight_scale": f.get_tensor(scale_key).cpu().contiguous(),
                    "comfy_quant": f.get_tensor(qkey).cpu().contiguous(),
                }
                if source_format == "nvfp4":
                    packed["weight_scale_2"] = f.get_tensor(scale2_key).cpu().contiguous()
                self.packed[name] = packed
                # CPU FP16 avoids a full extra BF16 model copy on GPU.
                self.residuals[name] = torch.zeros(tuple(module.weight.shape), device="cpu", dtype=torch.float16)
        if not self.modules:
            raise RuntimeError("No supported quantized Linear layers from the selected checkpoint matched the Anima model.")
        self.last_change_ratio = 0.0
        print("=" * 56)
        print(f"INFO: Projected {_anima_quant_choice_from_format(self.format).upper()} repair enabled.")
        print(f"- Quantized source: {self.source_path}")
        print(f"- Packed layers: {len(self.modules):,}")
        print("INFO: Optimizer updates are projected into stored codes/scales after every step.")
        print("INFO: Saves write those exact packed tensors without re-quantization.")
        print("INFO: Error-feedback residuals are CPU-offloaded in FP16 to reduce VRAM.")
        print("=" * 56)

    @torch.no_grad()
    def project_after_step(self):
        from scripts.convert_anima_to_quants import comfy_quant_info_tensor, dequantize_nvfp4_tensor, quantize_nvfp4_tensor, scaled_quant_tensor
        changed = total = 0
        for name, module in self.modules.items():
            residual_gpu = self.residuals[name].to(device=module.weight.device, dtype=module.weight.dtype)
            candidate = module.weight.detach() + residual_gpu
            if self.format == "nvfp4":
                qweight, qscale, qscale2 = quantize_nvfp4_tensor(candidate, scale_multiplier=self.scale_multiplier)
                projected = dequantize_nvfp4_tensor(qweight, qscale, qscale2, int(module.out_features), int(module.in_features), module.weight.dtype, module.weight.device)
            else:
                storage_dtype = {"int8_tensorwise": torch.int8, "float8_e4m3fn": torch.float8_e4m3fn, "float8_e5m2": torch.float8_e5m2}[self.format]
                qweight, qscale = scaled_quant_tensor(candidate, storage_dtype, self.format)
                projected = (qweight.to(device=module.weight.device).float() * qscale.to(device=module.weight.device).float()).to(module.weight.dtype)
            previous = self.packed[name]["weight"]
            # Compare encoded bytes. Direct comparison between INT8 and FP8 asks
            # PyTorch to promote the dtypes, which float8 intentionally does not
            # support. All projected formats here use one byte per stored code.
            if tuple(qweight.shape) == tuple(previous.shape):
                current_codes = qweight.detach().cpu().contiguous().view(torch.uint8)
                previous_codes = previous.detach().cpu().contiguous().view(torch.uint8)
                changed += int(torch.count_nonzero(current_codes.ne(previous_codes)).item())
            else:
                changed += qweight.numel()
            total += qweight.numel()
            residual_cpu = (candidate - projected).to(device="cpu", dtype=torch.float16)
            self.residuals[name].copy_(residual_cpu)
            module.weight.copy_(projected)
            self.packed[name]["weight"] = qweight
            self.packed[name]["weight_scale"] = qscale
            self.packed[name]["comfy_quant"] = comfy_quant_info_tensor(self.format)
            if self.format == "nvfp4":
                self.packed[name]["weight_scale_2"] = qscale2
            else:
                self.packed[name].pop("weight_scale_2", None)
            del residual_gpu, residual_cpu, candidate, projected
        self.last_change_ratio = changed / max(total, 1)
        return self.last_change_ratio

    def state_dict(self):
        return {"format": self.format, "scale_multiplier": self.scale_multiplier, "residuals": {n: r.detach().cpu() for n, r in self.residuals.items()}}

    def load_state_dict(self, state):
        if not state:
            return
        if state.get("format") != self.format:
            raise ValueError(f"Projected quant format changed on resume: {state.get('format')} -> {self.format}")
        for name, value in state.get("residuals", {}).items():
            if name in self.residuals and tuple(value.shape) == tuple(self.residuals[name].shape):
                self.residuals[name].copy_(value.to(self.residuals[name].device, self.residuals[name].dtype))

    def save_quantized(self, output_path, dit, key_prefix=""):
        from scripts.convert_anima_to_quants import comfy_quant_key_for_weight, comfy_scale2_key_for_weight, comfy_scale_key_for_weight
        output_path = Path(output_path)
        key_prefix = "" if str(key_prefix).lower() == "auto" else str(key_prefix or "")
        print(f"SAVE [1/4] Preparing exact packed checkpoint: {output_path}", flush=True)
        prepare_started = time.time()
        packed_by_state_key = {f"{name}.weight": self.packed[name] for name in self.modules}
        records = []
        for state_key, tensor in dit.state_dict().items():
            out_key = f"{key_prefix}{state_key}"
            packed = packed_by_state_key.get(state_key)
            if packed is not None:
                weight_dtype = torch.uint8 if self.format == "nvfp4" else {
                    "int8_tensorwise": torch.int8,
                    "float8_e4m3fn": torch.float8_e4m3fn,
                    "float8_e5m2": torch.float8_e5m2,
                }[self.format]
                scale_dtype = torch.float8_e4m3fn if self.format == "nvfp4" else torch.float32
                records.extend(((out_key, packed["weight"], weight_dtype),
                                (comfy_scale_key_for_weight(out_key), packed["weight_scale"], scale_dtype)))
                if self.format == "nvfp4":
                    records.append((comfy_scale2_key_for_weight(out_key), packed["weight_scale_2"], torch.float32))
                records.append((comfy_quant_key_for_weight(out_key), packed["comfy_quant"], torch.uint8))
            else:
                saved = tensor.detach().cpu().contiguous()
                records.append((out_key, saved, saved.dtype))
        with safe_open(str(self.source_path), framework="pt", device="cpu") as f:
            metadata = dict(f.metadata() or {})
        metadata["aozora_quantization"] = f"projected_{self.format}_repair"
        metadata["aozora_nvfp4_scale_multiplier"] = str(self.scale_multiplier)
        print(f"SAVE [1/4] Prepared {len(records):,} tensors in {time.time()-prepare_started:.1f}s.", flush=True)
        save_quant_records_with_progress(output_path, records, metadata)
        print(f"SAVE [4/4] Verifying {len(self.weight_keys):,} packed layers...", flush=True)
        with safe_open(str(output_path), framework="pt", device="cpu") as f:
            for verify_index, (name, weight_key) in enumerate(self.weight_keys.items(), start=1):
                out_weight_key = f"{key_prefix}{name}.weight"
                if not torch.equal(f.get_tensor(out_weight_key), self.packed[name]["weight"]):
                    raise RuntimeError(f"Packed save verification failed for {out_weight_key}")
                if verify_index % 50 == 0 or verify_index == len(self.weight_keys):
                    print(f"SAVE [4/4] Verified {verify_index:,}/{len(self.weight_keys):,} packed layers.", flush=True)
        print(f"SAVE COMPLETE in {time.time()-prepare_started:.1f}s: {output_path}", flush=True)
        return output_path


class AnimaBFloat16RepairController:
    """Full-precision control run using the otherwise identical repair objective."""
    format = "bfloat16"

    def __init__(self, dit, config):
        self.compute_dtype = config.compute_dtype
        print("=" * 56)
        print("INFO: BF16 repair-control training enabled.")
        print("INFO: Semantic/line-art and flow-matching losses remain identical to quant repair.")
        print("INFO: Optimizer updates are not projected into quantized codes.")
        print("=" * 56)

    @torch.no_grad()
    def project_after_step(self):
        return 0.0

    def state_dict(self):
        return {"format": self.format}

    def load_state_dict(self, state):
        if state and state.get("format", self.format) != self.format:
            raise ValueError(f"Repair format changed on resume: {state.get('format')} -> {self.format}")

    def save_quantized(self, output_path, dit, key_prefix=""):
        # Keep the common controller interface used by periodic/final saves.
        return save_dit_model(output_path, dit, torch.bfloat16, key_prefix, streaming_save=True)


def enable_anima_quantized_forward_repair(dit, config):
    if _anima_qat_target_format(config) == "bfloat16":
        return AnimaBFloat16RepairController(dit, config)
    return AnimaProjectedQuantController(dit, config)
SAFETENSORS_DTYPE_NAMES = {
    torch.float64: "F64",
    torch.float32: "F32",
    torch.float16: "F16",
    torch.bfloat16: "BF16",
    torch.int64: "I64",
    torch.int32: "I32",
    torch.int16: "I16",
    torch.int8: "I8",
    torch.uint8: "U8",
    torch.bool: "BOOL",
}
for _name, _code in (
    ("float8_e4m3fn", "F8_E4M3"),
    ("float8_e4m3fnuz", "F8_E4M3FNUZ"),
    ("float8_e5m2", "F8_E5M2"),
    ("float8_e5m2fnuz", "F8_E5M2FNUZ"),
    ("complex64", "C64"),
    ("uint64", "U64"),
    ("uint32", "U32"),
    ("uint16", "U16"),
):
    _dtype = getattr(torch, _name, None)
    if _dtype is not None:
        SAFETENSORS_DTYPE_NAMES[_dtype] = _code


def _safetensors_dtype_name(dtype):
    name = SAFETENSORS_DTYPE_NAMES.get(dtype)
    if name is None:
        raise TypeError(f"Cannot save tensor with unsupported safetensors dtype: {dtype}")
    return name


def _dtype_element_size(dtype):
    return torch.empty((), dtype=dtype).element_size()


def _stream_tensor_to_file(handle, tensor):
    tensor.reshape(-1).view(torch.uint8).numpy().tofile(handle)


def save_quant_records_with_progress(output_path, records, metadata):
    """Metadata-preserving streaming save with visible byte progress."""
    output_path = Path(output_path)
    tmp_path = output_path.with_name(f".{output_path.name}.{uuid.uuid4().hex}.tmp")
    header = {"__metadata__": {str(k): str(v) for k, v in metadata.items()}}
    offset = 0
    record_sizes = []
    for key, tensor, save_dtype in records:
        nbytes = tensor.numel() * _dtype_element_size(save_dtype)
        record_sizes.append(nbytes)
        header[key] = {"dtype": _safetensors_dtype_name(save_dtype), "shape": list(tensor.shape), "data_offsets": [offset, offset + nbytes]}
        offset += nbytes
    total_bytes = offset
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    header_bytes += b" " * ((8 - len(header_bytes) % 8) % 8)
    started = time.time()
    written = 0
    next_report = 0.05
    print(f"SAVE [2/4] Writing {len(records):,} tensors / {total_bytes / (1024**2):.1f} MiB to temporary file...", flush=True)
    try:
        with open(tmp_path, "wb") as handle:
            handle.write(struct.pack("<Q", len(header_bytes)))
            handle.write(header_bytes)
            for index, ((_, source_tensor, save_dtype), nbytes) in enumerate(zip(records, record_sizes), start=1):
                tensor_cpu = source_tensor.detach().to(device="cpu", dtype=save_dtype).contiguous() if torch.is_floating_point(source_tensor) else source_tensor.detach().cpu().contiguous()
                _stream_tensor_to_file(handle, tensor_cpu)
                del tensor_cpu
                written += nbytes
                fraction = written / max(total_bytes, 1)
                if fraction >= next_report or index == len(records):
                    elapsed = time.time() - started
                    rate = written / max(elapsed, 1e-6) / (1024**2)
                    print(f"SAVE [2/4] {fraction*100:5.1f}% ({written/(1024**2):.1f}/{total_bytes/(1024**2):.1f} MiB, {rate:.1f} MiB/s)", flush=True)
                    next_report = math.floor(fraction / 0.05 + 1.0) * 0.05
        print("SAVE [3/4] Finalizing atomic file replacement...", flush=True)
        os.replace(tmp_path, output_path)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise
    finally:
        gc.collect()


def shutdown_reporter_with_progress(reporter):
    finished = threading.Event()
    failure = []
    def worker():
        try:
            reporter.shutdown()
        except BaseException as exc:
            failure.append(exc)
        finally:
            finished.set()
    print("SHUTDOWN [1/2] Waiting for asynchronous reporter to finish...", flush=True)
    started = time.time()
    thread = threading.Thread(target=worker, name="repair-reporter-shutdown", daemon=True)
    thread.start()
    while not finished.wait(15.0):
        print(f"SHUTDOWN [1/2] Reporter still finishing ({time.time()-started:.0f}s elapsed)...", flush=True)
    thread.join()
    if failure:
        raise failure[0]
    print(f"SHUTDOWN [1/2] Reporter finished in {time.time()-started:.1f}s.", flush=True)

def save_safetensors_streaming(output_path, tensor_records):
    output_path = Path(output_path)
    tmp_path = output_path.with_name(f".{output_path.name}.{uuid.uuid4().hex}.tmp")
    header = {}
    offset = 0
    for key, source_tensor, save_dtype in tensor_records:
        nbytes = source_tensor.numel() * _dtype_element_size(save_dtype)
        header[key] = {
            "dtype": _safetensors_dtype_name(save_dtype),
            "shape": list(source_tensor.shape),
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes

    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    header_bytes += b" " * ((8 - (len(header_bytes) % 8)) % 8)

    try:
        with open(tmp_path, "wb") as f:
            f.write(struct.pack("<Q", len(header_bytes)))
            f.write(header_bytes)
            for _, source_tensor, save_dtype in tensor_records:
                tensor_cpu = (
                    source_tensor.detach().to(device="cpu", dtype=save_dtype).contiguous()
                    if torch.is_floating_point(source_tensor)
                    else source_tensor.detach().cpu().contiguous()
                )
                _stream_tensor_to_file(f, tensor_cpu)
                del tensor_cpu
        os.replace(tmp_path, output_path)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise



def _anima_quant_choice_from_format(format_name):
    mapping = {
        "nvfp4": "nvfp4",
        "int8_tensorwise": "int8",
        "float8_e4m3fn": "e4m3",
        "float8_e5m2": "e5m2",
    }
    return mapping[str(format_name)]


def _anima_keep_dtype_name(dtype):
    if dtype == torch.float32:
        return "float32"
    if dtype == torch.float16:
        return "float16"
    return "bfloat16"


def save_dit_model(output_path, dit, compute_dtype, key_prefix="", streaming_save=True):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nINFO: Saving DiT to: {output_path.name}")

    named_params = dict(dit.named_parameters())
    named_buffers = dict(dit.named_buffers())
    param_keys = set(named_params.keys())
    buffer_keys = set(named_buffers.keys())
    total_param_count = sum(p.numel() for p in named_params.values())
    trainable_param_count = sum(p.numel() for p in named_params.values() if p.requires_grad)
    frozen_param_count = total_param_count - trainable_param_count

    key_prefix = "" if str(key_prefix).lower() == "auto" else str(key_prefix or "")
    raw_state = dit.state_dict()
    tensor_records = []
    saved_param_count = 0
    saved_trainable_param_count = 0
    saved_frozen_param_count = 0
    saved_buffer_count = 0
    saved_bytes = 0
    dtype_counts = defaultdict(lambda: {"tensors": 0, "elements": 0})
    for k, v in raw_state.items():
        out_key = f"{key_prefix}{k}"
        save_dtype = compute_dtype if torch.is_floating_point(v) else v.dtype
        _safetensors_dtype_name(save_dtype)
        tensor_records.append((out_key, v, save_dtype))
        saved_bytes += v.numel() * _dtype_element_size(save_dtype)
        dtype_key = str(save_dtype).replace("torch.", "")
        dtype_counts[dtype_key]["tensors"] += 1
        dtype_counts[dtype_key]["elements"] += v.numel()
        if k in param_keys:
            saved_param_count += v.numel()
            if named_params[k].requires_grad:
                saved_trainable_param_count += v.numel()
            else:
                saved_frozen_param_count += v.numel()
        elif k in buffer_keys:
            saved_buffer_count += v.numel()

    missing_params = sorted(param_keys - set(raw_state.keys()))
    missing_trainable_params = [k for k in missing_params if named_params[k].requires_grad]
    extra_state_keys = sorted(set(raw_state.keys()) - param_keys - buffer_keys)

    print("INFO: DiT save accounting:")
    print(f"- Model parameters:       {total_param_count:,}")
    print(f"- Parameters saved:       {saved_param_count:,} / {total_param_count:,} ({(saved_param_count / max(total_param_count, 1)) * 100:.2f}%)")
    print(f"- Trainable saved:        {saved_trainable_param_count:,} / {trainable_param_count:,} ({(saved_trainable_param_count / max(trainable_param_count, 1)) * 100:.2f}%)")
    print(f"- Frozen saved:           {saved_frozen_param_count:,} / {frozen_param_count:,} ({(saved_frozen_param_count / max(frozen_param_count, 1)) * 100:.2f}%)")
    print(f"- Buffer elements saved:  {saved_buffer_count:,}")
    print(f"- State tensors saved:    {len(tensor_records):,}")
    print(f"- Estimated tensor bytes: {saved_bytes / (1024 ** 2):.2f} MiB")
    if dtype_counts:
        dtype_summary = ", ".join(
            f"{dtype}: {info['tensors']:,} tensors / {info['elements']:,} elems"
            for dtype, info in sorted(dtype_counts.items())
        )
        print(f"- Saved dtypes:           {dtype_summary}")
    if missing_params:
        print(f"WARNING: {len(missing_params):,} named parameters were not present in state_dict().")
        if missing_trainable_params:
            print(f"WARNING: {len(missing_trainable_params):,} missing parameters were trainable.")
        for name in missing_params[:10]:
            print(f"  -> missing param: {name}")
        if len(missing_params) > 10:
            print(f"  ... and {len(missing_params) - 10:,} more")
    else:
        print("INFO: All named DiT parameters are present in the saved state.")
    if extra_state_keys:
        print(f"INFO: State dict has {len(extra_state_keys):,} non-parameter/non-buffer tensor keys.")
        for name in extra_state_keys[:5]:
            print(f"  -> extra state: {name}")
        if len(extra_state_keys) > 5:
            print(f"  ... and {len(extra_state_keys) - 5:,} more")
    if key_prefix:
        print(f"INFO: Saved DiT keys with prefix: {key_prefix}")
    if streaming_save:
        print("INFO: Using streaming safetensors save.")
        save_safetensors_streaming(output_path, tensor_records)
    else:
        print("INFO: Using legacy safetensors save.")
        state = {}
        for out_key, source_tensor, save_dtype in tensor_records:
            tensor_cpu = (
                source_tensor.detach().to(device="cpu", dtype=save_dtype).contiguous()
                if torch.is_floating_point(source_tensor)
                else source_tensor.detach().cpu().contiguous()
            )
            state[out_key] = tensor_cpu
        save_file(state, str(output_path))
        del state
    del tensor_records
    gc.collect()
    file_size = output_path.stat().st_size if output_path.exists() else 0
    try:
        with safe_open(str(output_path), framework="pt", device="cpu") as f:
            disk_keys = set(f.keys())
        expected_keys = {f"{key_prefix}{k}" for k in raw_state.keys()}
        missing_disk_keys = sorted(expected_keys - disk_keys)
        unexpected_disk_keys = sorted(disk_keys - expected_keys)
        print(f"INFO: Safetensors file keys: {len(disk_keys):,} / expected {len(expected_keys):,}")
        print(f"INFO: Safetensors file size: {file_size / (1024 ** 2):.2f} MiB")
        if missing_disk_keys or unexpected_disk_keys:
            print(
                "WARNING: Safetensors key verification mismatch "
                f"(missing={len(missing_disk_keys):,}, unexpected={len(unexpected_disk_keys):,})."
            )
            for name in missing_disk_keys[:10]:
                print(f"  -> missing on disk: {name}")
            for name in unexpected_disk_keys[:10]:
                print(f"  -> unexpected on disk: {name}")
        else:
            print("INFO: Safetensors key verification passed.")
    except Exception as e:
        print(f"WARNING: Could not verify saved safetensors keys: {e}")
    print("INFO: DiT save complete.")
    return output_path


def save_anima_checkpoint_pt(global_step, micro_step, dit, optimizer, lr_scheduler, sampler, config, noise_generator=None, batch_sampler=None, quant_controller=None):
    output_dir = Path(config.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_filename = f"{Path(config.DIT_PATH).stem}_step_{global_step}.safetensors"
    state_filename = f"training_state_step_{global_step}.pt"
    if quant_controller is None:
        raise RuntimeError("Projected quant controller is required for repair checkpoint saving.")
    saved_model_path = quant_controller.save_quantized(
        output_dir / model_filename, dit, getattr(config, "ANIMA_DIT_SAVE_PREFIX", ""),
    )
    print("STATE [1/2] Collecting optimizer and projected-residual state...", flush=True)
    state_started = time.time()
    optim_state = optimizer.save_cpu_state() if hasattr(optimizer, "save_cpu_state") else optimizer.state_dict()
    print(f"STATE [1/2] State collected in {time.time()-state_started:.1f}s; writing training-state file...", flush=True)
    torch.save({
        "global_step": global_step,
        "micro_step": micro_step,
        "optimizer_state": optim_state,
        "projected_quant_state": quant_controller.state_dict() if quant_controller is not None else None,
        "sampler_seed": sampler.seed,
        "sampler_pool_index": sampler.pool_index,
        "timestep_sampler_state": sampler.state_dict() if hasattr(sampler, "state_dict") else None,
        "batch_sampler_epoch": max(batch_sampler.epoch - 1, 0) if batch_sampler is not None else 0,
        "random_state": random.getstate(),
        "numpy_state": np.random.get_state(),
        "torch_cpu_state": torch.get_rng_state(),
        "torch_cuda_state": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        "noise_generator_state": noise_generator.get_state() if noise_generator is not None else None,
    }, output_dir / state_filename)
    print(f"STATE [2/2] Training state saved in {time.time()-state_started:.1f}s: {output_dir / state_filename}", flush=True)


_ANIMA_LINEAR_SCHEDULE_CACHE = {}


def anima_ticket_to_sigma_timestep(ticket_indices, dtype):
    cache_key = (ticket_indices.device, dtype)
    if cache_key not in _ANIMA_LINEAR_SCHEDULE_CACHE:
        sigmas = torch.linspace(1.0, 0.0, 1001, device=ticket_indices.device)[:-1]
        _ANIMA_LINEAR_SCHEDULE_CACHE[cache_key] = (sigmas.to(dtype), (sigmas * 1000).to(dtype))
    sigmas, timesteps = _ANIMA_LINEAR_SCHEDULE_CACHE[cache_key]
    schedule_indices = 999 - ticket_indices
    return sigmas[schedule_indices], timesteps[schedule_indices]


def run_dit_forward(dit, noisy_latents, timesteps, prompt_emb, t5xxl_ids, config):
    model_output = dit(
        x=noisy_latents.unsqueeze(2),
        timesteps=timesteps / 1000,
        context=prompt_emb,
        t5xxl_ids=t5xxl_ids,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
    )
    return model_output.squeeze(2)


def flowmatch_noise_and_target(input_latents, noise, sigmas):
    sigmas = sigmas.view((sigmas.shape[0],) + (1,) * (input_latents.ndim - 1))
    return (1 - sigmas) * input_latents + sigmas * noise, noise - input_latents


def weighted_flowmatch_mse(model_pred, training_target, weights, spatial_mask=None, spatial_strength=0.5):
    squared_error = (model_pred.float() - training_target.float()).pow(2)
    if spatial_mask is not None and spatial_strength > 0.0:
        mask = spatial_mask.to(device=squared_error.device, dtype=squared_error.dtype)
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        if mask.shape[-2:] != squared_error.shape[-2:]:
            mask = F.interpolate(mask, size=squared_error.shape[-2:], mode="area")
        squared_error = squared_error * (1.0 + mask.clamp(0.0, 1.0) * spatial_strength)
    per_sample_loss = squared_error.flatten(1).mean(dim=1)
    return (per_sample_loss * weights.float()).mean()


def run_anima_dit_training(config):
    normalize_anima_config(config)

    is_resuming = getattr(config, "RESUME_TRAINING", False)
    required_paths = ["VAE_PATH", "TEXT_ENCODER_PATH"]
    required_paths.extend(["ANIMA_RESUME_MODEL_PATH", "ANIMA_RESUME_STATE_PATH"] if is_resuming else ["DIT_PATH"])
    for attr in required_paths:
        value = getattr(config, attr, "")
        if not value or not Path(value).exists():
            raise FileNotFoundError(f"{attr} is required for Anima DiT training: {value}")

    if config.SEED:
        set_seed(config.SEED)

    output_dir = Path(config.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    force_save_flag = Path(__file__).resolve().with_name("force_save.flag")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    micro_step = optimizer_step = 0
    optimizer_state = None
    projected_quant_state = None
    saved_sampler_pool_index = None
    saved_timestep_sampler_state = None
    saved_noise_generator_state = None
    saved_batch_sampler_epoch = 0
    if is_resuming:
        print("\n" + "=" * 50 + "\n--- RESUMING ANIMA DIT TRAINING SESSION ---\n")
        training_state = torch.load(Path(config.ANIMA_RESUME_STATE_PATH), map_location="cpu", weights_only=False)
        micro_step = training_state.get("micro_step", training_state.get("global_step", 0) * config.GRADIENT_ACCUMULATION_STEPS)
        optimizer_step = micro_step // config.GRADIENT_ACCUMULATION_STEPS
        optimizer_state = training_state.get("optimizer_state")
        projected_quant_state = training_state.get("projected_quant_state")
        saved_sampler_pool_index = training_state.get("sampler_pool_index")
        saved_timestep_sampler_state = training_state.get("timestep_sampler_state")
        saved_noise_generator_state = training_state.get("noise_generator_state")
        saved_batch_sampler_epoch = training_state.get("batch_sampler_epoch", 0)
        config.DIT_PATH = config.ANIMA_RESUME_MODEL_PATH
        if "random_state" in training_state:
            random.setstate(training_state["random_state"])
        if "numpy_state" in training_state:
            np.random.set_state(training_state["numpy_state"])
        if "torch_cpu_state" in training_state:
            torch.set_rng_state(training_state["torch_cpu_state"])
        if "torch_cuda_state" in training_state and training_state["torch_cuda_state"] is not None:
            torch.cuda.set_rng_state(training_state["torch_cuda_state"])
    else:
        print("\n" + "=" * 50 + "\n--- STARTING ANIMA DIT TRAINING ---\n" + "=" * 50 + "\n")

    print("INFO: Loading Anima pipeline components on CPU...")
    pipe = load_anima_pipe(config, torch.device("cpu"))

    print("INFO: Precomputing Anima DiT cache if needed...")
    precompute_and_cache_anima(config, pipe, device)

    print("INFO: Moving VAE/Text Encoder off GPU for DiT training...")
    pipe.vae.cpu()
    pipe.text_encoder.cpu()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    dit = pipe.dit

    dataset = AnimaCachedDataset(config)
    print(f"INFO: Cached Anima DiT dataset items: {len(dataset)}")
    print_anima_dataset_resolution_sample(dataset, sample_count=5)
    total_scheduler_timesteps = 1000
    timestep_sampler = AnimaTimestepSampler(config, total_scheduler_timesteps)
    if saved_timestep_sampler_state is not None:
        timestep_sampler.load_state_dict(saved_timestep_sampler_state)
    elif saved_sampler_pool_index is not None:
        timestep_sampler.pool_index = int(saved_sampler_pool_index) % len(timestep_sampler.ticket_pool)
    elif micro_step > 0:
        timestep_sampler.set_current_step(micro_step)

    image_schedule = build_image_batch_schedule(
        dataset,
        config.MAX_TRAIN_STEPS,
        config.BATCH_SIZE,
        config.SEED if config.SEED else 42,
        timestep_sampler.ticket_pool,
        timestep_sampler.bin_ranges,
        bool(getattr(config, "TIMESTEP_FORCE_IMAGE_BIN_SPREAD", False)),
    )
    image_schedule = pack_anima_sample_schedule(image_schedule, config.BATCH_SIZE)
    batch_sampler = PrecomputedImageBatchSampler(image_schedule, config.SEED if config.SEED else 42, micro_step if is_resuming else 0)
    print(f"INFO: Precomputed Anima image batch schedule for {len(image_schedule):,} step(s).")
    print("INFO: Anima dataset conditioning is keyed by seed and absolute sample position.")
    dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        collate_fn=anima_collate_fn,
    )
    if is_resuming and len(dataloader) <= 0:
        print("WARNING: Resume requested but Anima dataloader is empty; starting without batch skipping.")

    timestep_loss_weights = timestep_loss_curve_from_config(
        config,
        total_scheduler_timesteps,
        device=device,
        dtype=torch.float32,
    )

    reporter = AsyncReporter(total_steps=config.MAX_TRAIN_STEPS, test_param_name="dit")
    diagnostics = TrainingDiagnostics(config.GRADIENT_ACCUMULATION_STEPS)
    generator = torch.Generator(device=device)
    generator.manual_seed(config.SEED if config.SEED else 42)
    if saved_noise_generator_state is not None:
        generator.set_state(saved_noise_generator_state)

    print("INFO: Loading quantized-forward Anima student for repair training...")
    dit.to(device=device, dtype=config.compute_dtype)
    dit.train()
    freeze_dit_layers(dit, getattr(config, "DIT_EXCLUDE_TARGETS", []))
    quant_controller = enable_anima_quantized_forward_repair(dit, config)
    quant_controller.load_state_dict(projected_quant_state)

    params_to_optimize = [{"params": [p for p in dit.parameters() if p.requires_grad], "lr_scale": 1.0}]
    all_trainable_params = [p for group in params_to_optimize for p in group["params"]]
    optimizer = create_optimizer(config, params_to_optimize)
    lr_scheduler = CustomCurveLRScheduler(
        optimizer=optimizer,
        curve_points=config.LR_CUSTOM_CURVE,
        total_micro_steps=config.MAX_TRAIN_STEPS,
    )
    if optimizer_state:
        try:
            optimizer.load_cpu_state(optimizer_state) if hasattr(optimizer, "load_cpu_state") else optimizer.load_state_dict(optimizer_state)
        except Exception as e:
            print(f"WARNING: Could not load optimizer state: {e}")
        lr_scheduler.step(micro_step)

    print("INFO: Repair objective: original diffusion flow-matching target.")
    lineart_loss_enabled = ANIMA_REPAIR_LINEART_LOSS_ENABLED
    lineart_loss_strength = max(0.0, float(ANIMA_REPAIR_LINEART_LOSS_STRENGTH))
    print(f"INFO: Cached line-art loss: enabled={lineart_loss_enabled}, max_weight={1.0 + lineart_loss_strength:g}x")

    accumulated_latent_paths = set()
    optim_step_times = deque(maxlen=20)
    global_step_times = deque(maxlen=50)
    training_start_time = time.time()
    last_step_time = time.time()
    last_optim_step_log_time = time.time()
    optimizer.zero_grad(set_to_none=True)
    dataloader_iter = iter(dataloader)

    while micro_step < config.MAX_TRAIN_STEPS:
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            batch = next(dataloader_iter)
        if not batch:
            continue

        micro_step += 1
        step_start = time.time()
        lr_scheduler.step(micro_step)
        input_latents = batch["latents"].to(device=device, dtype=config.compute_dtype, non_blocking=True)
        prompt_emb = batch["prompt_emb"].to(device=device, dtype=config.compute_dtype, non_blocking=True)
        t5xxl_ids = batch["t5xxl_ids"].to(device=device, non_blocking=True)

        batch_size = input_latents.shape[0]
        ticket_indices, timestep_str = timestep_sampler.sample(batch_size)
        ticket_indices = ticket_indices.to(device=device)
        sigmas, timesteps = anima_ticket_to_sigma_timestep(ticket_indices, config.compute_dtype)
        loss_weights = timestep_loss_weights[ticket_indices]
        noise = torch.randn(input_latents.shape, device=device, dtype=config.compute_dtype, generator=generator)

        noisy_latents, training_target = flowmatch_noise_and_target(input_latents, noise, sigmas)
        model_pred = run_dit_forward(dit, noisy_latents, timesteps, prompt_emb, t5xxl_ids, config)
        loss = weighted_flowmatch_mse(
            model_pred, training_target, loss_weights,
            spatial_mask=batch.get("lineart_mask") if lineart_loss_enabled else None,
            spatial_strength=lineart_loss_strength,
        )
        raw_loss_value = loss.detach().float().item()
        (loss / config.GRADIENT_ACCUMULATION_STEPS).backward()

        diagnostics.step(raw_loss_value)
        accumulated_latent_paths.update(batch["latent_path"])
        diag_data_to_log = None

        if micro_step % config.GRADIENT_ACCUMULATION_STEPS == 0:
            raw_grad_norm = (
                optimizer.clip_grad_norm(config.CLIP_GRAD_NORM if config.CLIP_GRAD_NORM > 0 else float("inf"))
                if isinstance(optimizer, TitanAdamW)
                else torch.nn.utils.clip_grad_norm_(
                    all_trainable_params,
                    config.CLIP_GRAD_NORM if config.CLIP_GRAD_NORM > 0 else float("inf"),
                )
            )
            if isinstance(raw_grad_norm, torch.Tensor):
                raw_grad_norm = raw_grad_norm.item()
            clipped_grad_norm = min(raw_grad_norm, config.CLIP_GRAD_NORM) if config.CLIP_GRAD_NORM > 0 else raw_grad_norm

            optimizer.step()
            quant_change_ratio = quant_controller.project_after_step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1
            optim_step_time = time.time() - last_optim_step_log_time
            optim_step_times.append(optim_step_time)
            last_optim_step_log_time = time.time()

            diag_data_to_log = {
                "optim_step": optimizer_step,
                "avg_loss": diagnostics.get_average_loss(),
                "current_lr": optimizer.param_groups[-1]["lr"],
                "raw_grad_norm": raw_grad_norm,
                "clipped_grad_norm": clipped_grad_norm,
                "update_delta": 1.0 if raw_grad_norm > 0 else 0.0,
                "quant_change_ratio": quant_change_ratio,
                "optim_step_time": optim_step_time,
                "avg_optim_step_time": sum(optim_step_times) / len(optim_step_times),
            }
            diagnostics.reset()
            accumulated_latent_paths.clear()

            scheduled_save = (
                config.SAVE_EVERY_N_STEPS > 0
                and optimizer_step > 0
                and optimizer_step % config.SAVE_EVERY_N_STEPS == 0
            )
            force_save = consume_force_save_flag(force_save_flag)
            if scheduled_save or force_save:
                reason = "Emergency checkpoint requested" if force_save and not scheduled_save else "Saving checkpoint"
                reporter.log_message(f"\n--- {reason} at optimizer step {optimizer_step} ---")
                save_anima_checkpoint_pt(
                    optimizer_step,
                    micro_step,
                    dit,
                    optimizer,
                    lr_scheduler,
                    timestep_sampler,
                    config,
                    generator,
                    batch_sampler,
                    quant_controller,
                )

        step_duration = time.time() - last_step_time
        global_step_times.append(step_duration)
        last_step_time = time.time()
        avg_step_time = sum(global_step_times) / len(global_step_times) if global_step_times else 0.0
        reporter.log_step(
            micro_step,
            timing_data={
                "raw_step_time": time.time() - step_start,
                "elapsed_time": time.time() - training_start_time,
                "eta": (config.MAX_TRAIN_STEPS - micro_step) * avg_step_time,
                "loss": raw_loss_value,
                "timestep": int(timestep_str),
                "sigma": float(sigmas[0].detach().float().item()),
            },
            diag_data=diag_data_to_log,
        )

    reporter.log_message("\nTraining complete.")
    shutdown_reporter_with_progress(reporter)
    final_path = output_dir / f"{Path(config.DIT_PATH).stem}_trained_full_{str(uuid.uuid4())[:4]}.safetensors"
    saved_final_path = quant_controller.save_quantized(
        final_path, dit, getattr(config, "ANIMA_DIT_SAVE_PREFIX", ""),
    )
    print("All tasks complete. Final DiT saved.")


def main():
    from train import TrainingConfig

    config = TrainingConfig()
    run_anima_dit_training(config)


if __name__ == "__main__":
    main()
