# Aozora Trainer

Aozora Trainer is a GUI-based model fine-tuning trainer focused on making full-model training practical on single consumer GPUs.

It is built for users who want more control than a preset-only trainer, but do not want to manage large JSON configs by hand. The goal is simple: train SDXL UNet and Anima DiT models efficiently on 12 GB class GPUs while keeping the important training controls visible, editable, and understandable.

Aozora is for users willing to trade some speed for control—and for the ability
to train more of the model on hardware they already own.

> [**Currently supports**]
> - SDXL UNet models: epsilon (EPS), v-prediction (VPRED), and rectified flow
> - Anima DiT models

> [!WARNING]
> **Beta Software:** This project is still in active development. It is functional and currently trains successfully on my system, but it has not been broadly tested across different hardware, drivers, models, or environments.
>
> Current testing shows SDXL training can run with **80-90% of the full UNet trainable** on **11.80 GB VRAM**, at roughly **1.55 seconds per iteration** and **10-15 seconds per optimizer step**.
>
> Expect rough edges, bugs, and setup-specific issues. The main goal is to make SDXL training practical on **12 GB VRAM GPUs** without forcing low-resolution training or hiding every important option behind config files.

<table>
  <tr>
    <td><img src="https://i.ibb.co/Gf9BD0xw/NVIDIA-Overlay-C9-Yc5h5yx-Z.png" alt="GUI 1" width="425"/></td>
    <td><img src="https://i.ibb.co/9m6rQwZr/NVIDIA-Overlay-i-DUf-L7-Abgv.png" alt="GUI 2" width="425"/></td>
  </tr>
  <tr>
    <td><img src="https://i.ibb.co/3y0jM6qY/NVIDIA-Overlay-q-HAt-CFim-Uk.png" alt="GUI 3" width="425"/></td>
    <td><img src="https://i.ibb.co/YB7sk73K/NVIDIA-Overlay-igy-UYQ74i-W.png" alt="GUI 4" width="425"/></td>
  </tr>
</table>

---

## Quick Start

### Requirements

- latest NVIDIA GPU driver
- Python 3.11
- Git for Windows
- NVIDIA GPU with 12 GB or more VRAM
- 32 GB system RAM
- at least 20 GB free disk space, plus room for model caches
- Windows 10/11

CUDA Toolkit 12.x through 13.3 is recommended when building CUDA extensions or
wheels locally. Aozora's validated stack is CUDA 12.8. The standard installer
uses prebuilt CUDA 12.8 packages, so a separate Toolkit is normally unnecessary.

For local wheel builds on Windows, install Visual Studio 2022 Build Tools with
the **Desktop development with C++** workload, MSVC tools, and a Windows SDK.

### Windows Installation

Open Command Prompt or PowerShell and clone the repository:

```bat
git clone https://github.com/Hysocs/Aozora_Trainer.git
cd Aozora_Trainer
```

Then run setup from the same terminal:

```bat
setup.bat
```

This creates `portable_Venv`, installs PyTorch 2.9.1 with CUDA 12.8, and lets
you choose Flash Attention 2 or xFormers.

Alternatively, open the cloned `Aozora_Trainer` folder in File Explorer and
double-click `setup.bat`.

You can also download the repository as a ZIP from GitHub, extract it, and run
`setup.bat` from the extracted folder.

### Start the GUI

Run:

```bat
start_gui.bat
```

Select the model and dataset, choose the correct caption source, prediction
type, and VAE settings, review the learning-rate curve, then press **Train**.

### Recommended 12 GB VRAM Exclusions

SDXL **Exclude Layers (Keywords)**:

```text
down_blocks.2.*.transformer_blocks.*.ff.net.2.*, mid_block.*.transformer_blocks.*.ff.net.2.*, up_blocks.0.*.transformer_blocks.*.ff.net.2.*
```

Anima **Exclude DiT Layers**:

```text
llm_adapter.*
```

### Settings and How to Know if Training Is Working

- **SDXL VAE:** Match the Dataset-tab VAE settings to the base model and rebuild
  the cache after changes. Incorrect settings cause artifacts; use a preset if unsure.
- **Learning rate:** LR controls update strength. Start around `7.5e-6` and
  increase cautiously in `2.5e-6` increments.
- **Timesteps:** These are training noise levels. Use the uniform preset unless
  you have a reason to focus on a specific range.
- **Loss weighting:** This changes the importance of each timestep. Keep it flat
  at `1.0` unless you understand what the model needs.

Before a long run, test **20-50 images** for **10-20 epochs** and compare saved
checkpoints with fixed generation settings. A healthy run learns gradually
without damaging the base model. Fast artifacts usually mean excessive LR or
incorrect VAE settings; little change suggests low LR, weak captions, or too
little training.


### Dataset

Place images and optional matching captions in one folder:

```text
dataset/
  image_001.png
  image_001.txt
  image_002.png
  image_002.txt
```

Captions must share the image's base name. Missing captions fall back to the
filename. Supported images: `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`.

#### Recommended JSON Captions

For more flexible training, select `json` as the GUI caption type and place a
matching `.json` file beside every image:

```text
dataset/
  image_001.png
  image_001.json
```

```json
{
  "tags": "score_9, score_8_up, source_anime, 1girl, solo, pink_hair",
  "nl": "An anime-style girl with pink hair is sitting in a bedroom.",
  "tags_nl": "score_9, score_8_up, source_anime, 1girl, solo, pink_hair. An anime-style girl with pink hair is sitting in a bedroom.",
  "nl_tags": "An anime-style girl with pink hair is sitting in a bedroom. score_9, score_8_up, source_anime, 1girl, solo, pink_hair"
}
```

The four keys provide tags, natural language, tags followed by natural language,
and natural language followed by tags. Filling all four is recommended. Aozora
caches every non-empty variant and uses the four GUI percentage controls to
choose between them during training.


### Command-line Training

After running `setup.bat`, you can train directly:

```bat
REM SDXL UNet
portable_Venv\Scripts\python.exe train.py --config "C:\path\to\config.json"

REM Anima DiT
portable_Venv\Scripts\python.exe train_anima.py --config "C:\path\to\config.json"
```

---

## Feature Overview

### Low-VRAM Full-Model Training

Aozora keeps full-model training practical on consumer GPUs through:

- CPU-offloaded optimizer states
- mixed precision training
- gradient checkpointing
- memory-efficient attention backends
- latent and text-embedding caching
- selective layer targeting

### GUI-Based Training Workflow

The GUI exposes datasets, model and VAE paths, prediction type, batching,
precision, optimizers, learning-rate and timestep graphs, layer exclusions,
caching, checkpointing, and resume settings.

### Custom Optimizers

#### Raven

Raven is the balanced default. It stores optimizer state on the CPU and uses
shared GPU buffers for FP32 update math.

#### Titan

Titan offloads gradients as well as optimizer state, reducing VRAM further at
the cost of slower updates.

### Visual Training Controls

- **Timestep allocation:** Shape training coverage with uniform, Wave,
  Logit-Normal, or Beta distributions.
- **Timestep loss weights:** Use a separate graph to control how strongly errors
  from each timestep range affect training.
- **Learning-rate curve:** Draw warmup, hold, decay, or custom schedules.
- **Layer targeting:** Freeze matching UNet or DiT modules to reduce memory use
  and control which parts are trained.

### Caching and Resolution

- Cache VAE latents, text embeddings, pooled embeddings, and null conditioning.
- Encode long SDXL captions across multiple CLIP chunks.
- Assign images to aspect-ratio-aware buckets with correct SDXL size metadata.
- Optionally cache nearby bucket variants for better aspect-ratio coverage.

Multi-bucket and caption-chunk caching improve flexibility but use more disk
space and preprocessing time.

### Model Compatibility

The SDXL UNet trainer supports `epsilon`, `v_prediction`, and `rectified_flow`
models. The separate Anima trainer targets Anima DiT models. Model paths,
prediction type, VAE normalization, and conditioning components must match the
selected model family.

### Monitoring, Checkpoints, and Resume

The GUI reports loss, timestep, learning rate, gradient norm, timing, ETA, VRAM,
and checkpoint status. Saved training states include optimizer, scheduler, and
timestep-sampler state so interrupted runs can resume.

## Notes and Limitations

- Standard SDXL UNet and Anima DiT `.safetensors` checkpoints are the main targets.
- Merged or unusual architectures may need different layer exclusions.

---

## Current Goals

Aozora grew from the effort to make full-model SDXL training reliable on a
single consumer GPU. Earlier experiments showed how easily hidden defaults,
memory limits, and model-specific settings could derail a run, so the project
focuses on making those choices visible and understandable.

Current priorities are stable low-VRAM training, safer defaults, clearer
diagnostics, fewer hidden assumptions, and practical support for SDXL UNet and
Anima DiT models.
New features should strengthen that workflow without turning the trainer into a
collection of opaque presets.

