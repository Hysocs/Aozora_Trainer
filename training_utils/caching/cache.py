import os
import random
import re
from pathlib import Path

import torch


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
CAPTION_JSON_TYPES = ("tags", "nl", "tags_nl", "nl_tags")
CAPTION_JSON_PRIMARY_TYPE = "tags_nl"
CAPTION_JSON_VARIANT_RE = re.compile(r"_json_(tags|nl|tags_nl|nl_tags)$")
CACHE_INDEX_NAME = "dataset_index.pt"
CACHE_IMAGE_LAYOUT_OPTION_KEYS = (
    "cache_schema_version",
    "bucket_layout",
    "max_bucket_resolution",
    "bucket_resolution_tiers",
    "should_upscale",
    "caption_source_type",
)
CACHE_TEXT_OPTION_KEYS = (
    "cache_schema_version",
    "text_cache_float_dtype",
    "caption_source_type",
    "caption_json_types",
    "caption_chunking_enabled",
    "caption_embedding_layout",
)
CACHE_LATENT_OPTION_KEYS = (
    "cache_schema_version",
    "vae_cache_float_dtype",
    "vae_normalization_mode",
    "vae_shift_factor",
    "vae_scaling_factor",
    "vae_latent_channels",
    "vae_path",
    "vae_source_path",
    "vae_source_size",
    "vae_source_mtime_ns",
)


def cache_options_match_for_keys(cached_options, expected_options, keys):
    if not isinstance(cached_options, dict) or not isinstance(expected_options, dict):
        return False
    return all(cached_options.get(key) == expected_options.get(key) for key in keys)


def cache_image_layout_options_match(cached_options, expected_options, extra_keys=()):
    return cache_options_match_for_keys(
        cached_options,
        expected_options,
        CACHE_IMAGE_LAYOUT_OPTION_KEYS + tuple(extra_keys or ()),
    )


def cache_text_options_match(cached_options, expected_options, extra_keys=()):
    return cache_options_match_for_keys(
        cached_options,
        expected_options,
        CACHE_TEXT_OPTION_KEYS + tuple(extra_keys or ()),
    )


def cache_latent_options_match(cached_options, expected_options, extra_keys=()):
    return cache_options_match_for_keys(
        cached_options,
        expected_options,
        CACHE_LATENT_OPTION_KEYS + tuple(extra_keys or ()),
    )


def cache_index_path(cache_dir):
    return Path(cache_dir) / CACHE_INDEX_NAME


def cache_index_exists(cache_dir):
    return cache_index_path(cache_dir).exists()


def load_cache_index(cache_dir_or_path):
    path = Path(cache_dir_or_path)
    if path.is_dir():
        path = cache_index_path(path)

    return torch.load(path, map_location="cpu", weights_only=False)


def save_cache_index(cache_dir, payload):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_index_path(cache_dir)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)
    return path


def collect_image_paths(root):
    root = Path(root)
    return sorted(
        (
            p
            for ext in IMAGE_EXTENSIONS
            for p in root.rglob(f"*{ext}")
        ),
        key=lambda p: p.relative_to(root).as_posix().casefold(),
    )


def stable_cache_item_key(item):
    """Order cached image/bucket variants independently of filesystem traversal."""
    target_size = tuple(item.get("target_size", (0, 0)))
    return (
        str(item.get("relative_path", item.get("image_key", ""))).replace("\\", "/").casefold(),
        int(item.get("bucket_variant_index", 0) or 0),
        target_size,
        str(item.get("lat_path", item.get("te_path", ""))).replace("\\", "/").casefold(),
    )


def file_stat_signature(path):
    path = Path(path)
    if not path.exists():
        return {"exists": False, "path": str(path)}
    stat = path.stat()
    return {
        "exists": True,
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def image_file_signature(image_path):
    return file_stat_signature(image_path)


def caption_sidecar_path(image_path, caption_mode="txt"):
    image_path = Path(image_path)
    return image_path.with_suffix(".json" if caption_source_type(caption_mode) == "json" else ".txt")


def caption_file_signature_for_image(image_path, caption_mode="txt"):
    sidecar = caption_sidecar_path(image_path, caption_mode)
    signature = file_stat_signature(sidecar)
    signature["mode"] = caption_source_type(caption_mode)
    return signature


def cached_file_signatures_match(item, image_path, caption_mode):
    image_sig = item.get("image_file_signature")
    caption_sig = item.get("caption_file_signature")
    if not image_sig or not caption_sig:
        return None
    return (
        image_sig == image_file_signature(image_path)
        and caption_sig == caption_file_signature_for_image(image_path, caption_mode)
    )


def cache_stem_for_image(root, image_path):
    return str(Path(image_path).relative_to(root).with_suffix("")).replace(os.sep, "_")


def cache_item_stem_from_te_path(path):
    name = Path(path).name
    if not name.endswith("_te.pt"):
        return None
    stem = name[:-len("_te.pt")]
    return strip_json_caption_suffix(stem)


def cache_base_stem_from_te_path(path):
    stem = cache_item_stem_from_te_path(path)
    if stem is None:
        return None
    return re.sub(r"_mb\d+$", "", stem)


def cache_base_stem_from_cache_path(path):
    path = Path(path)
    if path.name.endswith("_te.pt"):
        return cache_base_stem_from_te_path(path)
    if path.name.endswith("_lat.pt"):
        stem = path.name[:-len("_lat.pt")]
        return re.sub(r"_mb\d+$", "", stem)
    return None


def strip_json_caption_suffix(stem):
    return CAPTION_JSON_VARIANT_RE.sub("", str(stem))


def json_caption_cache_suffix(caption_type, enabled=True):
    return f"_json_{caption_type}" if enabled else ""


def caption_types_for_cache(json_caption_mode):
    return CAPTION_JSON_TYPES if json_caption_mode else ("txt",)


def caption_source_type(config_or_value=None):
    value = config_or_value
    if config_or_value is not None and not isinstance(config_or_value, str):
        value = getattr(config_or_value, "CAPTION_SOURCE_TYPE", "txt")
    value = str(value or "txt").strip().lower()
    return "json" if value == "json" else "txt"


def json_caption_mode_enabled(config_or_value=None):
    return caption_source_type(config_or_value) == "json"


def choose_caption_variant(rng, weights):
    total = sum(max(0, int(weights.get(k, 0) or 0)) for k in CAPTION_JSON_TYPES)
    if total <= 0:
        return CAPTION_JSON_PRIMARY_TYPE
    roll = rng.uniform(0, total)
    upto = 0
    for key in CAPTION_JSON_TYPES:
        upto += max(0, int(weights.get(key, 0) or 0))
        if roll <= upto:
            return key
    return CAPTION_JSON_PRIMARY_TYPE


def caption_variant_index(variant_paths, path_key="te_path"):
    return {
        key: {path_key: str(variant_paths[key])}
        for key in CAPTION_JSON_TYPES
        if key in variant_paths
    }


def selected_caption_variant_path(item, rng, weights, path_key="te_path", primary_key="te_path", enabled=True):
    variants = item.get("caption_variants")
    if enabled and isinstance(variants, dict):
        available_weights = {key: weights.get(key, 0) for key in variants}
        caption_type = choose_caption_variant(rng, available_weights)
        variant = variants.get(caption_type) or variants.get(CAPTION_JSON_PRIMARY_TYPE) or next(iter(variants.values()))
        if isinstance(variant, dict) and variant.get(path_key):
            return variant[path_key]
    return item.get(primary_key)


def lat_path_for_te_path(te_path):
    te_path = Path(te_path)
    name = te_path.name
    if not name.endswith("_te.pt"):
        return Path(str(te_path).replace("_te.pt", "_lat.pt"))
    stem = name[:-len("_te.pt")]
    stem = strip_json_caption_suffix(stem)
    return te_path.with_name(f"{stem}_lat.pt")


def text_cache_paths_for_index_item(item, path_key="te_path", primary_key="te_path"):
    variants = item.get("caption_variants")
    if isinstance(variants, dict):
        return [
            value.get(path_key)
            for value in variants.values()
            if isinstance(value, dict) and value.get(path_key)
        ]
    path = item.get(primary_key)
    return [path] if path else []


def cache_payload_options_match_for_index_item(item, expected_options, cache_paths_for_item, text_options_match, latent_options_match):
    for cache_path in cache_paths_for_item(item):
        cache_path = Path(cache_path)
        if not cache_path.exists():
            return False
        try:
            payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        except Exception:
            return False
        if cache_path.name.endswith("_te.pt"):
            if not text_options_match(payload.get("cache_options"), expected_options):
                return False
        elif cache_path.name.endswith("_lat.pt"):
            if not isinstance(payload, dict):
                return False
            if not latent_options_match(payload.get("cache_options"), expected_options):
                return False
    return True


def cache_payload_options_match_for_index(index_data, expected_options, cache_paths_for_item, text_options_match, latent_options_match):
    if not isinstance(index_data, dict):
        return False
    return all(
        cache_payload_options_match_for_index_item(
            item,
            expected_options,
            cache_paths_for_item,
            text_options_match,
            latent_options_match,
        )
        for item in index_data.get("files", [])
    )


def te_paths_for_index_item(item):
    return text_cache_paths_for_index_item(item, path_key="te_path", primary_key="te_path")


def remove_cache_pair_for_te_path(te_path):
    te_path = Path(te_path)
    lat_path = lat_path_for_te_path(te_path)
    for path in (te_path, lat_path):
        try:
            if path.exists():
                path.unlink()
        except OSError as e:
            print(f"WARNING: Could not remove stale cache file {path}: {e}")


def remove_cache_files_for_stem(cache_dir, base_stem):
    name_re = re.compile(
        rf"^{re.escape(str(base_stem))}"
        rf"(?:_mb\d+)?"
        rf"(?:_json_(?:{'|'.join(CAPTION_JSON_TYPES)}))?"
        rf"_(?:te|lat)\.pt$"
    )
    for path in Path(cache_dir).glob("*.pt"):
        if not name_re.match(path.name):
            continue
        try:
            path.unlink()
        except OSError as e:
            print(f"WARNING: Could not remove stale cache file {path}: {e}")


def expected_cache_paths_for_metadata(root, cache_dir, meta, caption_types, json_caption_mode):
    safe_filename = cache_stem_for_image(root, meta["ip"]) + meta.get("cache_suffix", "")
    text_paths = {
        caption_type: Path(cache_dir) / f"{safe_filename}{json_caption_cache_suffix(caption_type, json_caption_mode)}_te.pt"
        for caption_type in caption_types
    }
    return text_paths, Path(cache_dir) / f"{safe_filename}_lat.pt"


def cache_metadata_matches(payload, root, meta):
    if not isinstance(payload, dict):
        return False
    return (
        payload.get("relative_path") == str(meta["ip"].relative_to(root))
        and tuple(payload.get("original_size", ())) == tuple(meta["original_size"])
        and tuple(payload.get("scaled_size", payload.get("original_size", ()))) == tuple(meta.get("scaled_size", meta["original_size"]))
        and tuple(payload.get("target_size", ())) == tuple(meta["target_resolution"])
        and tuple(payload.get("crop_coords", (0, 0))) == tuple(meta.get("crop_coords", (0, 0)))
        and int(payload.get("bucket_variant_index", 0) or 0) == int(meta.get("bucket_variant_index", 0) or 0)
    )
