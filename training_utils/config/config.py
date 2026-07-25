import copy

# ====================================================================================
# DEFAULT CONFIGURATION
# ====================================================================================

CONFIG_VERSION = 5
MODE_SDXL = "sdxl"
MODE_ANIMA = "anima"
TRAINING_MODE_SDXL = "SDXL"
TRAINING_MODE_ANIMA = "Anima DiT"
MODE_LABELS = {
    MODE_SDXL: TRAINING_MODE_SDXL,
    MODE_ANIMA: TRAINING_MODE_ANIMA,
}

# --- Paths ---
SINGLE_FILE_CHECKPOINT_PATH = "./model.safetensors"
VAE_PATH = ""
OUTPUT_DIR = "./output"
OUTPUT_NAME = "auto"

# --- Architecture ---
TRAINING_MODE = "SDXL"
DIT_PATH = ""
DIT_VAE_PATH = ""
ANIMA_DIT_SAVE_PREFIX = "auto"
TEXT_ENCODER_PATH = ""
TOKENIZER_PATH = ""
TOKENIZER_T5XXL_PATH = ""

# --- Resume Training ---
RESUME_TRAINING = False
RESUME_MODEL_PATH = ""
RESUME_STATE_PATH = ""
ANIMA_RESUME_MODEL_PATH = ""
ANIMA_RESUME_STATE_PATH = ""

# --- Dataset Configuration ---
INSTANCE_DATASETS = [
    {
        "path": "./data",
        "repeats": 1,
    }
]

# --- Caching & Data Loaders ---
CACHING_BATCH_SIZE = 2
TEXT_CACHE_PRECISION = "bfloat16"
VAE_CACHE_PRECISION = "bfloat16"
NUM_WORKERS = 0
UNCONDITIONAL_DROPOUT = False
UNCONDITIONAL_DROPOUT_CHANCE = 0.0
QWEN_NULL_DROPOUT_CHANCE = 0.0
T5_NULL_DROPOUT_CHANCE = 0.0
TEXT_CONDITIONING_SCALE_ENABLED = False
TEXT_CONDITIONING_SCALE_MIN = 1.0
TEXT_CONDITIONING_SCALE_MAX = 1.0
T5_TOKEN_DROPOUT_ENABLED = False
T5_TOKEN_DROPOUT_CHANCE = 0.0
T5_TOKEN_DROPOUT_MIN = 0.0
T5_TOKEN_DROPOUT_MAX = 0.0
CAPTION_CHUNKING_ENABLED = False
CAPTION_SOURCE_TYPE = "txt"
CAPTION_TAGS_PERCENT = 40
CAPTION_NL_PERCENT = 10
CAPTION_TAGS_NL_PERCENT = 25
CAPTION_NL_TAGS_PERCENT = 25

# --- Aspect Ratio Bucketing ---
SHOULD_UPSCALE = False
MAX_BUCKET_RESOLUTION = 1024  # Options: 896, 1024, 1152, 1536
MULTI_BUCKET_ENABLED = False
MULTI_BUCKET_EXTRA_BUCKETS = 0
MAX_BUCKET_RESOLUTION_CHOICES = (896, 1024, 1152, 1536)

# --- Core Training Parameters ---
PREDICTION_TYPE = "v_prediction"
MAX_TRAIN_STEPS = 10000
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
MIXED_PRECISION = "bfloat16"
CLIP_GRAD_NORM = 1.0
SEED = 42
ANIMA_GRADIENT_CHECKPOINTING_MODE = "Full"
ANIMA_SEMANTIC_LOSS_ENABLED = False

# --- Saving ---
SAVE_EVERY_N_STEPS = 1000
ANIMA_STREAMING_SAVE = True

# --- UNet Layer Exclusion (Blacklist) ---
UNET_EXCLUDE_TARGETS = "conv1, conv2"
DIT_EXCLUDE_TARGETS = ""

# --- Learning Rate Scheduler ---
LR_CUSTOM_CURVE = [
    [0.0, 0.0],
    [0.05, 8.0e-7],
    [0.85, 8.0e-7],
    [1.0, 1.0e-7]
]
LEARNING_RATE = 8.0e-7
LR_GRAPH_MIN = 0.0
LR_GRAPH_MAX = 1.0e-6

# --- Timestep Allocation (Ticket System) ---
# "bin_size": resolution of the bars (e.g., 50 means 0-50, 50-100...)
# "counts": list of integers representing tickets per bin.
TIMESTEP_ALLOCATION = {
    "bin_size": 100,
    "counts": [] # Populated by GUI based on MAX_TRAIN_STEPS
}
TIMESTEP_STRATIFIED_SAMPLING = False
TIMESTEP_FORCE_IMAGE_BIN_SPREAD = False
TIMESTEP_LOSS_WEIGHT_CURVE = [
    [0.0, 1.0],
    [1.0, 1.0],
]

# --- Optimizer Configuration ---
OPTIMIZER_TYPE = "raven"
RAVEN_PARAMS = {
    "betas": [0.9, 0.999],
    "eps": 1e-8,
    "weight_decay": 0.01,
    "debias_strength": 1.0,
    "momentum_dtype": "bfloat16"
}
PAGED_ADAMW_8BIT_PARAMS = {
    "betas": [0.9, 0.999],
    "eps": 1e-8,
    "weight_decay": 0.01,
}
TITAN_PARAMS = {
    "betas": [0.9, 0.999],
    "eps": 1e-8,
    "weight_decay": 0.01,
    "debias_strength": 1.0,
    "momentum_dtype": "bfloat16"
}
# --- Loss Configuration ---
LOSS_TYPE = "MSE"

# --- Advanced & Miscellaneous ---
MEMORY_EFFICIENT_ATTENTION = "sdpa"
TIMESTEP_MODE = "Wave"
TIMESTEP_ODDS_SCALE = 3.0
# --- DiT / Anima Cache Configuration ---
ANIMA_CACHE_FOLDER_NAME = ".precomputed_anima_dit_cache"
VAE_CACHING_TILED = True
VAE_CACHING_TILE_SIZE = [96, 96]
VAE_CACHING_TILE_STRIDE = [72, 72]
REBUILD_CACHE = False

# --- VAE Configuration ---
VAE_NORMALIZATION_MODE = "scalar"  # scalar or flux_bn32 (ComfyUI Flux 32ch BN layout)
VAE_SHIFT_FACTOR = None      # None = auto-detect, else use this value
VAE_SCALING_FACTOR = None    # None = auto-detect, else use this value
VAE_LATENT_CHANNELS = None   # None = use model's channels, else force this


FLAT_KEYS = [
    "SINGLE_FILE_CHECKPOINT_PATH", "VAE_PATH", "OUTPUT_DIR", "OUTPUT_NAME", "TRAINING_MODE",
    "DIT_PATH", "DIT_VAE_PATH", "ANIMA_DIT_SAVE_PREFIX",
    "TEXT_ENCODER_PATH", "TOKENIZER_PATH", "TOKENIZER_T5XXL_PATH",
    "RESUME_TRAINING", "RESUME_MODEL_PATH", "RESUME_STATE_PATH",
    "ANIMA_RESUME_MODEL_PATH", "ANIMA_RESUME_STATE_PATH", "INSTANCE_DATASETS",
    "CACHING_BATCH_SIZE", "TEXT_CACHE_PRECISION", "VAE_CACHE_PRECISION",
    "NUM_WORKERS", "UNCONDITIONAL_DROPOUT",
    "UNCONDITIONAL_DROPOUT_CHANCE", "QWEN_NULL_DROPOUT_CHANCE",
    "T5_NULL_DROPOUT_CHANCE", "TEXT_CONDITIONING_SCALE_ENABLED",
    "TEXT_CONDITIONING_SCALE_MIN", "TEXT_CONDITIONING_SCALE_MAX",
    "T5_TOKEN_DROPOUT_ENABLED", "T5_TOKEN_DROPOUT_CHANCE",
    "T5_TOKEN_DROPOUT_MIN", "T5_TOKEN_DROPOUT_MAX",
    "CAPTION_CHUNKING_ENABLED", "CAPTION_SOURCE_TYPE", "CAPTION_TAGS_PERCENT",
    "CAPTION_NL_PERCENT", "CAPTION_TAGS_NL_PERCENT", "CAPTION_NL_TAGS_PERCENT",
    "SHOULD_UPSCALE", "MAX_BUCKET_RESOLUTION", "MULTI_BUCKET_ENABLED", "MULTI_BUCKET_EXTRA_BUCKETS",
    "PREDICTION_TYPE", "MAX_TRAIN_STEPS", "BATCH_SIZE",
    "GRADIENT_ACCUMULATION_STEPS", "MIXED_PRECISION", "CLIP_GRAD_NORM", "ANIMA_GRADIENT_CHECKPOINTING_MODE", "ANIMA_SEMANTIC_LOSS_ENABLED",
    "SEED", "SAVE_EVERY_N_STEPS", "ANIMA_STREAMING_SAVE", "UNET_EXCLUDE_TARGETS", "DIT_EXCLUDE_TARGETS",
    "LR_CUSTOM_CURVE", "LEARNING_RATE", "LR_GRAPH_MIN", "LR_GRAPH_MAX",
    "TIMESTEP_ALLOCATION", "TIMESTEP_STRATIFIED_SAMPLING", "TIMESTEP_FORCE_IMAGE_BIN_SPREAD", "TIMESTEP_LOSS_WEIGHT_CURVE", "OPTIMIZER_TYPE", "RAVEN_PARAMS", "PAGED_ADAMW_8BIT_PARAMS", "TITAN_PARAMS",
    "LOSS_TYPE",
    "MEMORY_EFFICIENT_ATTENTION", "TIMESTEP_MODE", "TIMESTEP_ODDS_SCALE",
    "ANIMA_CACHE_FOLDER_NAME", "VAE_CACHING_TILED", "VAE_CACHING_TILE_SIZE",
    "VAE_CACHING_TILE_STRIDE", "REBUILD_CACHE", "VAE_NORMALIZATION_MODE",
    "VAE_SHIFT_FACTOR", "VAE_SCALING_FACTOR", "VAE_LATENT_CHANNELS",
]

PER_MODE_FLAT_KEYS = [
    "OUTPUT_DIR", "OUTPUT_NAME", "RESUME_TRAINING", "INSTANCE_DATASETS", "CACHING_BATCH_SIZE",
    "TEXT_CACHE_PRECISION", "VAE_CACHE_PRECISION", "NUM_WORKERS",
    "UNCONDITIONAL_DROPOUT", "UNCONDITIONAL_DROPOUT_CHANCE",
    "QWEN_NULL_DROPOUT_CHANCE", "T5_NULL_DROPOUT_CHANCE",
    "TEXT_CONDITIONING_SCALE_ENABLED", "TEXT_CONDITIONING_SCALE_MIN",
    "TEXT_CONDITIONING_SCALE_MAX", "T5_TOKEN_DROPOUT_ENABLED",
    "T5_TOKEN_DROPOUT_CHANCE", "T5_TOKEN_DROPOUT_MIN",
    "T5_TOKEN_DROPOUT_MAX", "CAPTION_CHUNKING_ENABLED", "SHOULD_UPSCALE",
    "CAPTION_SOURCE_TYPE", "CAPTION_TAGS_PERCENT", "CAPTION_NL_PERCENT",
    "CAPTION_TAGS_NL_PERCENT", "CAPTION_NL_TAGS_PERCENT",
    "MAX_BUCKET_RESOLUTION", "MULTI_BUCKET_ENABLED",
    "MULTI_BUCKET_EXTRA_BUCKETS", "PREDICTION_TYPE", "MAX_TRAIN_STEPS",
    "BATCH_SIZE", "GRADIENT_ACCUMULATION_STEPS", "MIXED_PRECISION",
    "CLIP_GRAD_NORM", "SEED", "SAVE_EVERY_N_STEPS", "LR_CUSTOM_CURVE",
    "LEARNING_RATE", "LR_GRAPH_MIN", "LR_GRAPH_MAX", "TIMESTEP_ALLOCATION", "TIMESTEP_STRATIFIED_SAMPLING", "TIMESTEP_FORCE_IMAGE_BIN_SPREAD", "TIMESTEP_LOSS_WEIGHT_CURVE",
    "OPTIMIZER_TYPE", "RAVEN_PARAMS", "PAGED_ADAMW_8BIT_PARAMS", "TITAN_PARAMS",
    "LOSS_TYPE", "MEMORY_EFFICIENT_ATTENTION", "TIMESTEP_MODE", "TIMESTEP_ODDS_SCALE",
    "VAE_NORMALIZATION_MODE", "VAE_SHIFT_FACTOR", "VAE_SCALING_FACTOR",
    "VAE_LATENT_CHANNELS", "REBUILD_CACHE",
]

MODE_SPECIFIC_FLAT_KEYS = {
    MODE_SDXL: [
        "SINGLE_FILE_CHECKPOINT_PATH", "VAE_PATH", "RESUME_MODEL_PATH",
        "RESUME_STATE_PATH", "UNET_EXCLUDE_TARGETS",
    ],
    MODE_ANIMA: [
        "DIT_PATH", "DIT_VAE_PATH", "ANIMA_DIT_SAVE_PREFIX", "ANIMA_STREAMING_SAVE",
        "TEXT_ENCODER_PATH", "TOKENIZER_PATH", "TOKENIZER_T5XXL_PATH",
        "ANIMA_RESUME_MODEL_PATH", "ANIMA_RESUME_STATE_PATH",
        "DIT_EXCLUDE_TARGETS", "ANIMA_CACHE_FOLDER_NAME", "ANIMA_GRADIENT_CHECKPOINTING_MODE",
        "ANIMA_SEMANTIC_LOSS_ENABLED",
        "VAE_CACHING_TILED", "VAE_CACHING_TILE_SIZE", "VAE_CACHING_TILE_STRIDE",
    ],
}

NESTED_NAME_OVERRIDES = {
    "SINGLE_FILE_CHECKPOINT_PATH": "base_model_path",
    "DIT_PATH": "dit_model_path",
    "DIT_VAE_PATH": "vae_path",
    "ANIMA_DIT_SAVE_PREFIX": "dit_save_prefix",
    "TOKENIZER_PATH": "qwen_tokenizer",
    "TOKENIZER_T5XXL_PATH": "t5xxl_tokenizer",
    "RESUME_TRAINING": "resume_training",
    "RESUME_MODEL_PATH": "resume_model_path",
    "RESUME_STATE_PATH": "resume_state_path",
    "ANIMA_RESUME_MODEL_PATH": "resume_model_path",
    "ANIMA_RESUME_STATE_PATH": "resume_state_path",
}


def mode_key_from_label(value):
    text = str(value or "").strip().lower()
    if text in {MODE_ANIMA, TRAINING_MODE_ANIMA.lower()} or text.startswith("anima"):
        return MODE_ANIMA
    return MODE_SDXL


def nested_key_for(mode_key, flat_key):
    suffix = NESTED_NAME_OVERRIDES.get(flat_key, flat_key.lower())
    if suffix.startswith(f"{mode_key}_"):
        return suffix
    return f"{mode_key}_{suffix}"


def flat_defaults():
    return {key: copy.deepcopy(globals()[key]) for key in FLAT_KEYS if key in globals()}


def mode_flat_keys(mode_key):
    return [*PER_MODE_FLAT_KEYS, *MODE_SPECIFIC_FLAT_KEYS.get(mode_key, [])]


def default_mode_config(mode_key):
    defaults = flat_defaults()
    config = {
        nested_key_for(mode_key, flat_key): copy.deepcopy(defaults.get(flat_key))
        for flat_key in mode_flat_keys(mode_key)
    }
    return config


def default_preset():
    return {
        "config_version": CONFIG_VERSION,
        "active_mode": MODE_SDXL,
        MODE_SDXL: default_mode_config(MODE_SDXL),
        MODE_ANIMA: default_mode_config(MODE_ANIMA),
    }


def nest_flat_config(flat_config, mode_key=None, base_preset=None):
    flat_config = copy.deepcopy(flat_config)
    mode_key = mode_key_from_label(mode_key or flat_config.get("TRAINING_MODE"))
    preset = copy.deepcopy(base_preset) if base_preset else default_preset()
    preset["config_version"] = CONFIG_VERSION
    preset["active_mode"] = mode_key
    preset.setdefault(mode_key, default_mode_config(mode_key))
    for flat_key in mode_flat_keys(mode_key):
        if flat_key in flat_config:
            preset[mode_key][nested_key_for(mode_key, flat_key)] = copy.deepcopy(flat_config[flat_key])
    return preset


def normalize_preset(config_data):
    if not isinstance(config_data, dict):
        return default_preset()
    preset = default_preset()
    preset["active_mode"] = mode_key_from_label(config_data.get("active_mode"))
    for mode_key in (MODE_SDXL, MODE_ANIMA):
        if isinstance(config_data.get(mode_key), dict):
            valid_keys = {
                nested_key_for(mode_key, flat_key)
                for flat_key in mode_flat_keys(mode_key)
            }
            legacy_timestep_loss_key = f"{mode_key}_use_timestep_loss_weight"
            curve_key = nested_key_for(mode_key, "TIMESTEP_LOSS_WEIGHT_CURVE")
            odds_scale_key = nested_key_for(mode_key, "TIMESTEP_ODDS_SCALE")
            legacy_odds_keys = [f"{mode_key}_timestep_ticket_shift", f"{mode_key}_ticket_shift", f"{mode_key}_sigma_shift"]
            checkpoint_mode_key = nested_key_for(mode_key, "ANIMA_GRADIENT_CHECKPOINTING_MODE")
            if odds_scale_key not in config_data[mode_key]:
                for old_odds_key in legacy_odds_keys:
                    if old_odds_key in config_data[mode_key]:
                        preset[mode_key][odds_scale_key] = copy.deepcopy(config_data[mode_key][old_odds_key])
                        break
            if (
                config_data[mode_key].get(legacy_timestep_loss_key)
                and curve_key not in config_data[mode_key]
            ):
                preset[mode_key][curve_key] = {"preset": "bell"}
            preset[mode_key].update({
                key: copy.deepcopy(value)
                for key, value in config_data[mode_key].items()
                if key in valid_keys
            })
            if mode_key == MODE_ANIMA:
                checkpoint_mode = str(
                    preset[mode_key].get(checkpoint_mode_key, "Full")
                ).strip().title()
                preset[mode_key][checkpoint_mode_key] = (
                    checkpoint_mode
                    if checkpoint_mode in {"Full", "Conservative"}
                    else "Full"
                )
            timestep_mode_key = nested_key_for(mode_key, "TIMESTEP_MODE")
            if preset[mode_key].get(timestep_mode_key) == "Shift":
                preset[mode_key][timestep_mode_key] = "Odds-Scaled (Z-Image)"
    return preset


def flatten_preset(config_data, mode_key=None):
    preset = normalize_preset(config_data)
    mode_key = mode_key_from_label(mode_key or preset.get("active_mode"))
    flat = flat_defaults()
    flat["TRAINING_MODE"] = MODE_LABELS[mode_key]
    mode_block = preset.get(mode_key, {})
    for flat_key in mode_flat_keys(mode_key):
        nested_key = nested_key_for(mode_key, flat_key)
        if nested_key in mode_block:
            flat[flat_key] = copy.deepcopy(mode_block[nested_key])
    if mode_key == MODE_ANIMA:
        flat["VAE_PATH"] = flat.get("DIT_VAE_PATH", "")
        flat["RESUME_MODEL_PATH"] = ""
        flat["RESUME_STATE_PATH"] = ""
    return flat
