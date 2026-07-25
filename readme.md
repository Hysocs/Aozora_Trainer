# Aozora Trainer

Aozora Trainer is a GUI-based model fine-tuning trainer focused on making full-model training practical on single consumer GPUs.

It is built for users who want more control than a preset-only trainer, but do not want to manage large JSON configs by hand. The goal is simple: train SDXL models efficiently on 12 GB class GPUs while keeping the important training controls visible, editable, and understandable.

> [**Currently supports**]
> - SDXL (EPS, VPRED, RECT FLOW)
> - Anima

> [!WARNING]
> **Beta Software:** This project is still in active development. It is functional and currently trains successfully on my system, but it has not been broadly tested across different hardware, drivers, models, or environments.
>
> Current testing shows SDXL training can run with **80–90% of the full UNet trainable** on **11.80 GB VRAM**, at roughly **1.55 seconds per iteration** and **15 seconds per optimizer step**.
>
> Expect rough edges, bugs, and setup-specific issues. The main goal is to make SDXL training practical on **12 GB VRAM GPUs** without forcing low-resolution training or hiding every important option behind config files.
>
> **Note:** Earlier versions may have caused failed training runs. Stabilizing SDXL training on consumer hardware is difficult, and this project is still evolving.

<table>
  <tr>
    <td><img src="https://i.imgur.com/KWgV0OE.png" alt="GUI 1" width="425"/></td>
    <td><img src="https://i.imgur.com/XURoDBZ.png" alt="GUI 2" width="425"/></td>
  </tr>
  <tr>
    <td><img src="https://i.imgur.com/uwacboa.png" alt="GUI 3" width="425"/></td>
    <td><img src="https://i.imgur.com/1OzLZAu.png" alt="GUI 4" width="425"/></td>
  </tr>
</table>

---

## What This Trainer Does

Aozora is designed for SDXL fine-tuning on a single NVIDIA GPU.

It supports standard SDXL-style models and modern prediction types, including:

- `epsilon`
- `v_prediction`
- `rectified_flow`

This makes it usable with common SDXL model families as well as newer v-prediction and flow-matching models.

The trainer handles dataset preparation, caption embedding caching, latent caching, timestep sampling, optimizer configuration, checkpoint saving, resume states, and live GUI monitoring.

---

## Who This Is For

This trainer is mainly for users who:

- Have a **12 GB to 16 GB NVIDIA GPU** (8 GB can work buts its super slow)
- Want to train SDXL on a **single GPU**
- Want GUI controls instead of editing configs manually
- Want to train more of the UNet without needing a 24 GB GPU
- Want control over timesteps, learning rate shape, layer targeting, and optimizer behavior
- Are comfortable testing beta software and adjusting settings when needed

---

## Feature Overview

### Low-VRAM SDXL Training

Aozora is built around keeping SDXL training usable on consumer GPUs.

The trainer combines:

- CPU-offloaded optimizer states
- mixed precision training
- gradient checkpointing
- memory-efficient attention backends
- latent caching
- text embedding caching
- selective layer targeting
- checkpoint/resume support

This allows more of the SDXL UNet to remain trainable while keeping VRAM usage low.

---

### GUI-Based Training Workflow

Most major training options are exposed through the GUI.

You can configure:

- dataset folders
- base model path
- VAE settings
- prediction type
- batch size
- gradient accumulation
- learning rate curve
- timestep distribution
- optimizer type
- optimizer parameters
- precision mode
- checkpoint frequency
- resume paths
- layer exclusion keywords
- caption chunking
- multi-bucket caching

The goal is to make the training setup readable and editable without constantly opening config files.

---

### Custom Optimizers

Aozora includes custom optimizers designed around low-VRAM full-model training.

#### Raven

Raven is the balanced default optimizer.

It is designed to reduce VRAM usage while still behaving close to a full-precision AdamW-style optimizer. It keeps optimizer state off the GPU and uses shared GPU buffers during the update step.

Best for:

- general SDXL fine-tuning
- 12 GB class GPUs
- users who want the safest default option
- training a large part of the UNet

#### Titan

Titan is a more aggressive VRAM-saving optimizer option.

It is intended for cases where Raven is still too heavy or where you want to push trainable parameter count lower in memory usage. It may be slower, but can make larger training setups fit.

Best for:

- lower VRAM pressure
- heavier layer coverage
- experiments where speed matters less than fitting the run
  
---

### Timestep Ticket Allocation

Instead of relying only on basic random timestep sampling, Aozora uses a visual timestep ticket system.

You can control how much training is assigned to different timestep ranges. This is useful because different model types and training goals often benefit from different timestep focus.

The GUI includes distribution tools such as:

- Wave
- Logit-Normal
- Beta

This gives more direct control over where training effort is spent, especially for v-prediction and rectified-flow models.

---

### Visual Learning Rate Curve

Aozora includes a graph-based learning rate scheduler.

Instead of picking only a fixed scheduler preset, you can shape the LR curve visually. The trainer interpolates between curve points and applies that learning rate during training.

Useful for:

- warmup curves
- cosine-like decay
- linear decay
- custom ramps
- restart-style schedules
- experimental LR shapes

---

### Layer Targeting

The trainer supports excluding UNet layers by keyword.

This lets you reduce VRAM usage and control what parts of the model are updated. For example, you can freeze specific blocks or module types while keeping the rest trainable.

This is useful for:

- fitting larger runs into VRAM
- reducing overfitting
- testing which parts of the UNet matter most
- training most of the model while excluding the most expensive sections

---

### Latent and Text Embedding Caching

Aozora caches expensive preprocessing work before training.

It can cache:

- VAE latents
- SDXL text encoder embeddings
- pooled text embeddings
- null conditioning embeddings when needed

This reduces repeated work during training and helps keep training speed consistent.

---

### Caption Chunking

Caption chunking allows longer captions to be encoded in multiple CLIP-sized chunks instead of being cut down to a single 77-token window.

This is useful for datasets with long natural-language captions or detailed tag lists.

---

### Multi-Bucket Cache

The trainer can cache images into multiple nearby bucket resolutions.

This helps reduce aspect-ratio overfitting by exposing the model to nearby shape variations instead of tying each concept too strongly to one exact training resolution.

This can increase cache time and disk usage, but it is useful for concepts that need to generate well across more than one aspect ratio.

---

### Resolution-Aware Bucketing

Images are assigned to SDXL-style bucket resolutions based on their aspect ratio and target pixel area.

The trainer stores original size, target size, crop coordinates, and scaled size metadata so SDXL conditioning receives the correct size information during training.

---

### VAE Normalization Options

Aozora includes configurable VAE normalization support.

Supported modes include:

- scalar shift/scale normalization
- Flux BN32-style normalization when using compatible VAE sources

This is useful for models or VAEs that do not follow the same latent scaling assumptions as standard SDXL.

---

### Prediction Type Support

The trainer supports:

- epsilon prediction
- v-prediction
- rectified flow

This allows the same trainer to work across different SDXL model types instead of assuming every model uses the same training target.

---

### Training Console and Live Metrics

The GUI includes a training console and live metrics view.

The console is designed to handle large amounts of output without freezing the UI by storing logs in a compressed buffer and only rendering the visible section.

Training output includes:

- loss
- timestep
- step time
- ETA
- optimizer step
- learning rate
- gradient norm
- checkpoint status
- VRAM reporting

---

### Checkpointing and Resume

Aozora can save training checkpoints during a run and resume from saved state files.

This includes:

- model checkpoint output
- optimizer state
- scheduler state
- timestep sampler state
- resume model path
- resume state path

This is important for long training runs, unstable beta testing, or recovering from crashes.

---

## Quick Start

### Windows

1. Put your images in a dataset folder.
2. Add matching `.txt` caption files next to the images.
3. Run `setup.bat`.
4. Run `start_gui.bat`.
5. Select your base model.
6. Select your dataset folder.
7. Pick or adjust a preset.
8. **(Important)** set a vae config matching your model!
9. Press **Train**.

### Linux

Linux support expects an NVIDIA driver and a CUDA-compatible PyTorch installation.
The PySide6 GUI is not currently validated on Linux, so the configuration-file CLI
is the recommended route.

1. Create and activate a virtual environment (for example, `.venv`).
2. Install the PyTorch build appropriate for your CUDA/driver combination.
3. Run `python -m pip install -r requirements.txt`.
4. Continue with the command-line instructions below.

## Command-line training

Select or copy a JSON preset, then edit its model, dataset, tokenizer, and output
paths for the current machine. Use `train.py` for SDXL and `train_anima.py` for
Anima.

### Windows CLI

Run `setup.bat` first, then open a terminal in the repository:

```bat
REM SDXL
portable_Venv\Scripts\python.exe train.py --config "C:\path\to\config.json"

REM Anima
portable_Venv\Scripts\python.exe train_anima.py --config "C:\path\to\config.json"
```

### Linux CLI

Activate the environment created during Linux setup, then run:

```bash
# SDXL
python train.py --config /path/to/config.json

# Anima
python train_anima.py --config /path/to/config.json
```

Absolute paths are OS-specific. A config copied from Windows can still be opened on
Linux, but paths such as `C:\\models\\model.safetensors` must be selected again on
the Linux filesystem (and vice versa).

Checkpoints are saved to:

```txt
output/checkpoints/
```

If training crashes or you stop the run, resume from the latest saved `.pt` state file and matching model checkpoint.

---

## Dataset Format

Use a folder of images with optional matching captions.

Example:

```txt
dataset/
  image_001.png
  image_001.txt
  image_002.png
  image_002.txt
  image_003.jpg
  image_003.txt
```

If a caption file is missing, the image filename is used as the fallback caption.

Supported image formats include:

* `.jpg`
* `.jpeg`
* `.png`
* `.webp`
* `.bmp`
* `.tiff`

---

## Requirements

Recommended:

* Windows 10/11 or a modern Linux distribution
* Python 3.10
* NVIDIA GPU with 12 GB+ VRAM
* 32 GB system RAM
* 20 GB+ free disk space for the environment
* additional disk space for latent/text cache files
* recent NVIDIA drivers

The training code is OS-neutral, but Linux support has not yet received the same
amount of hardware testing as Windows. The provided `setup.bat` remains
Windows-only; Linux environments use the manual setup steps above.

A 12 GB GPU is the main target, but actual memory use depends on:

* model type
* optimizer
* trainable layer count
* resolution
* batch size
* gradient accumulation
* attention backend
* caption chunking
* multi-bucket cache settings

---

## Notes and Limitations

* This is beta software.
* Currently only tested mostly on my own system.
* Some hardware or driver setups may need changes.
* Standard SDXL-format `.safetensors` models are the main target.
* Merged models may behave differently depending on architecture changes, i recommend training only base models to avoid issue.
* Layer targeting depends on module names, so unusual model structures may need different exclude keywords.
* Flow models need correct prediction and shift settings.
* CPU-offloaded optimizers save VRAM but can be slower than fully GPU-resident optimizers.
* Multi-bucket caching improves flexibility but increases cache size.
* Caption chunking can improve long-caption handling but increases embedding size.

---

## Current Goal

The current goal of Aozora is not to be the most feature-packed trainer.

The goal is to make SDXL fine-tuning practical, inspectable, and efficient on consumer hardware while keeping the training process understandable.

Future improvements should keep that same direction:

* fewer hidden assumptions
* more visible controls
* stable low-VRAM behavior
* better defaults
* cleaner debugging
* practical full-model SDXL training on single GPUs

