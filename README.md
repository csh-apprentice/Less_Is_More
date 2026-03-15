<div align="center">
<h1>Less is More: Data-Efficient Adaptation for Controllable Text-to-Video Generation</h1>

<a href="https://arxiv.org/abs/2511.17844"><img src="https://img.shields.io/badge/arXiv-2511.17844-b31b1b" alt="arXiv"></a>
<a href="https://csh-apprentice.github.io/Less_Is_More/"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a>
<a href="https://huggingface.co/csh-apprentice/Less_Is_More"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Checkpoints-blue" alt="HuggingFace Checkpoints"></a>
<a href="https://huggingface.co/datasets/csh-apprentice/Less_Is_More-dataset"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue" alt="HuggingFace Dataset"></a>

**CVPR 2026**

[Shihan Cheng](https://www.isis.vanderbilt.edu/team/373/)<sup>1</sup>, [Nilesh Kulkarni](https://nileshkulkarni.github.io/)<sup>2</sup>, [David Hyde](https://dabh.io/)<sup>1</sup>, [Dmitriy Smirnov](https://dsmirnov.com/)<sup>2</sup>

**[Vanderbilt University](https://www.vanderbilt.edu/)<sup>1</sup>** &nbsp;&nbsp; **[Netflix](https://research.netflix.com/)<sup>2</sup>**

</div>

---

We present **Less Is More**, a method for injecting controllable camera properties (shutter speed, aperture/bokeh, color temperature) into a frozen text-to-video backbone (Wan2.1-T2V-14B) using a small scalar conditioning signal. Training requires only a tiny synthetic dataset and a few hours of compute.

---

## Method overview

The conditioning scalar (e.g., normalized shutter speed value) is injected into the **deepest third** of the Wan2.1 transformer blocks via lightweight FPS-adapter modules (LoRA-style, rank 32). A full-model LoRA is trained jointly. At inference, the **GRAFT** strategy applies adapters only to the deepest third, preserving backbone generation quality while enabling precise camera control.

| Property | Training dataset | Conditioning range |
|---|---|---|
| Shutter speed | `syn_shutter/` (synthetic, multi-fps videos) | normalized shutter value |
| Aperture (bokeh) | `syn_aperture/` (Blender synthetic) | normalized aperture value |
| Color temperature | `syn_temperature/` (synthetic images) | normalized Kelvin value |

---

## Requirements

- Linux (required — DeepSpeed does not support native Windows)
- CUDA 12.x driver (tested: driver 535 with CUDA 12.4 toolkit), Python 3.12
- For training: 2× A100 80GB (or equivalent)
- For inference: 1× GPU with ≥ 36 GB VRAM (A100 80GB recommended for full 49-frame generation; A6000 48GB sufficient for short smoke tests)

---

## Installation

```bash
git clone https://github.com/csh-apprentice/Less_Is_More.git
cd Less_Is_More
```

**One-step install** (creates the `scpipe` conda env):
```bash
# Optional: set your GPU architecture (default: 8.0 8.6 covers A100 + A6000/A40)
# H100 users: export TORCH_CUDA_ARCH_LIST="9.0"
bash install.sh
conda activate scpipe
```

`install.sh` handles the correct install order automatically. See the script for details.

<details>
<summary>Manual steps (if install.sh fails)</summary>

```bash
conda create -n scpipe python=3.12 -y && conda activate scpipe

# 1. PyTorch — must use pip wheel, NOT conda (conda causes MKL symbol conflict)
#    cu121 bundles its own CUDA 12.1 runtime; works on any driver >= 525
pip install torch==2.4.1+cu121 torchvision==0.19.1+cu121 torchaudio==2.4.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# 2. All other packages (versions are pinned in requirements.txt)
pip install -r requirements.txt

# 3. flash-attn — must be built from source after torch is installed
#    v2.8.1 requires nvcc >= 12.5; v2.7.2.post1 works with nvcc 12.x
export CUDA_HOME=/usr/local/cuda          # path to your CUDA toolkit
export TORCH_CUDA_ARCH_LIST="8.0 8.6"    # A100=8.0, A6000/A40=8.6, H100=9.0
export MAX_JOBS=8
pip install flash-attn==2.7.2.post1 --no-build-isolation
```
</details>

---

## Pretrained backbone

Download the Wan2.1-T2V-14B backbone from HuggingFace (~30 GB).

**With the scpipe environment active** (huggingface-cli is installed as a dependency):
```bash
conda activate scpipe
huggingface-cli download Wan-AI/Wan2.1-T2V-14B \
    --local-dir /path/to/Wan2.1-T2V-14B
```

Or use Python directly (no extra install needed):
```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Wan-AI/Wan2.1-T2V-14B', local_dir='/path/to/Wan2.1-T2V-14B')
"
```

Then update `ckpt_path` in any training config you use:

```toml
# configs/train_temperature.toml (and train_shutter.toml, train_bokeh.toml)
ckpt_path = '/path/to/Wan2.1-T2V-14B'
```

> For inference the backbone path is passed via `--config` (must match ckpt_path in the TOML)
> or patched at runtime. The smoke test does this automatically via a temp TOML.

---

## Pretrained adapters (paper checkpoints)

Download our trained adapters from HuggingFace:

> **[csh-apprentice/Less_Is_More](https://huggingface.co/csh-apprentice/Less_Is_More)**

Place them under `checkpoints/`:
```
checkpoints/
├── shutter/epoch1000/
│   ├── adapter_model.safetensors
│   └── adapter_config.json
├── temperature/epoch1000/
│   └── ...
└── bokeh/epoch1000/
    └── ...
```

---

## Dataset preparation

Datasets are small (<10 MB each) and fully synthetic. Generation scripts are provided in the original repository.

**Expected layout** (relative to repo root):
```
dataset/
├── syn_shutter/               # shutter speed
│   ├── 1s4f/videos/
│   ├── 1s8f/videos/
│   └── ... (9 durations × 3 frame counts)
├── syn_aperture/              # aperture (bokeh)
│   ├── 1s/
│   └── ... (5 durations)
└── syn_temperature/           # color temperature
    ├── 1s/
    └── ... (5 durations)
```

Dataset configs already reference `./dataset/...` relative paths — no editing needed once the data is placed correctly.

Download from HuggingFace: **[csh-apprentice/Less_Is_More-dataset](https://huggingface.co/datasets/csh-apprentice/Less_Is_More-dataset)**
```bash
huggingface-cli download csh-apprentice/Less_Is_More-dataset --repo-type dataset --local-dir dataset/
```

---

## Training

Training uses [DeepSpeed](https://github.com/microsoft/DeepSpeed) pipeline parallelism. Run from the repo root:

**Shutter speed:**
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
deepspeed --num_gpus=2 train.py --deepspeed \
    --config configs/train_shutter.toml
```

**Aperture (bokeh):**
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
deepspeed --num_gpus=2 train.py --deepspeed \
    --config configs/train_bokeh.toml
```

**Color temperature:**
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
deepspeed --num_gpus=2 train.py --deepspeed \
    --config configs/train_temperature.toml
```

Checkpoints are saved under `./output/<timestamp>/`. The relevant inference artifact is `epoch<N>/adapter_model.safetensors`.

### Multi-GPU configuration

**`--num_gpus` and `pipeline_stages`** are the two knobs:

| Setup | `--num_gpus` | `pipeline_stages` in TOML | Notes |
|---|---|---|---|
| 2× A100 80GB (paper) | `2` | `1` | Data-parallel ZeRO; each GPU holds the full model |
| 1× A100 80GB | `1` | `1` | Single-GPU; remove the `NCCL_*` env vars |
| 2× GPU, model too large for one | `2` | `2` | Pipeline-parallel; model split across both GPUs |

With `pipeline_stages = 1` (default), DeepSpeed uses ZeRO data parallelism — both GPUs train on different micro-batches and sync gradients. This is what the paper used.

**On A100 SXM4 (NVLink):** the `NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1` flags disable peer-to-peer and InfiniBand transport, which was required on PCIe-connected A6000s. On NVLink-enabled machines you can drop them for better GPU-to-GPU bandwidth:
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
deepspeed --num_gpus=2 train.py --deepspeed \
    --config configs/train_temperature.toml
```

**Resume from checkpoint:**
```bash
deepspeed --num_gpus=2 train.py --deepspeed \
    --config configs/train_temperature.toml \
    --resume_from_checkpoint
```
Resumes from the most recent DeepSpeed checkpoint in the run directory. Pass a specific subfolder name to resume from a particular step.

### Ablation training (deepest-third LoRA only)

To reproduce the ablation study (LoRA restricted to the deepest third of transformer blocks, matching the FPS adapter's block set):
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
deepspeed --num_gpus=2 train.py --deepspeed \
    --config configs/train_temperature_ablation.toml
```
At startup you should see `[ABLATION] lora_blocks=deepest_third: restricting LoRA to blocks 27–39 (13 of 40)` confirming the mode is active. Run inference on the resulting checkpoint with the standard script (no extra flags needed — the checkpoint naturally contains only deepest-third LoRA weights).

---

## Inference

All inference modes are controlled by `inference/test_fps_batch_prompts.py`. **Always run from the repo root** with `PYTHONPATH=.` so that `models/`, `utils/`, and `inference_utils/` are importable.

**GRAFT (recommended — paper's clean method):**
```bash
PYTHONPATH=. python inference/test_fps_batch_prompts.py \
    --config configs/train_temperature.toml \
    --checkpoint checkpoints/temperature/epoch1000 \
    --fps_values 0.2 0.5 0.8 \
    --steps 50 --frames 49 \
    --output_dir output/results/temperature \
    --prompt_file metric/high_quality_prompts_96.txt \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走." \
    --seed 42 --width 512 --height 512 \
    --graft \
    --port 29500
```

**Conditioning value reference:**
- `--fps_values` accepts one or more scalar values in the range the model was trained on
- Multiple values generate one video per value, useful for comparing across the control range

**Other inference modes:**

| Flag | Description |
|---|---|
| `--graft` | Paper's clean method: LoRA on all blocks, FPS adapter on deepest third only |
| *(none)* | Dirty method: apply everything everywhere |
| `--base_only` | Base LoRA only, no FPS adapter |
| `--fps_only` | FPS adapter only, no base LoRA |
| `--clean` | Original Wan2.1 backbone, no adapters |

---

## Evaluation

### FEP score (Fidelity-Enhanced Prompt following)

First generate baseline (clean backbone, `--steps 1 --frames 4`), then generate model outputs and compare:

```bash
python metric/video_score_calculator_extended.py \
    --orig_parent_dir output/onestep/clean_seed42 \
    --adapt_parent_dir output/onestep/temperature_epoch1000 \
    --output_file scores/temperature/fep_scores.json
```

### SVP score (Semantic Video Preservation)

Generate 50-step, 49-frame videos on the 96 evaluation prompts, then run:

```bash
# CLIP + X-CLIP score
python metric/xclip_score_calculator.py \
    --videos_dir output/50_49/temperature_epoch1000 \
    --prompt_file metric/high_quality_prompts_96.txt

# VQA score (place calculate_vqa_score.py inside utils/t2v_metrics/)
python metric/calculate_vqa_score.py \
    --videos_dir output/50_49/temperature_epoch1000 \
    --prompt_file metric/high_quality_prompts_96.txt
```

---

## Smoke test

To verify the installation is correct before full inference. **Run from the repo root with the scpipe environment active.**

```bash
# Tier 0: no GPU required (~10 seconds)
# Checks: import chain, TOML path-cleanliness, checkpoint key structure, PEFT config
# Requires: paper checkpoints placed under checkpoints/<run_id>/epoch1000/
bash smoke_test.sh

# Tier 1: GPU required (~40 min on A6000 for 96 prompts × 4 frames × 4 steps)
# Checks: clean backbone inference + GRAFT inference with 3 conditioning values
# Before running: edit BACKBONE_PATH at the top of smoke_test.sh
bash smoke_test.sh --tier1
```

Expected output: 4 `[PASS]` lines for Tier 0, 3 `[PASS]` lines for Tier 1, plus videos in `output/smoke_test/{clean,graft}/`.

> Tier 1 uses all 96 evaluation prompts for thorough coverage. On A6000 48GB this takes ~40 min
> with 4 frames/4 steps. On A100 80GB expect ~25 min.

---

## Citation

```bibtex
@inproceedings{cheng2026lessismore,
  title     = {Less is More: Data-Efficient Adaptation for Controllable Text-to-Video Generation},
  author    = {Cheng, Shihan and Kulkarni, Nilesh and Hyde, David and Smirnov, Dmitriy},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026}
}
```

---

## Acknowledgements

This work was conducted during an internship at Netflix. The codebase is built on top of [diffusion-pipe](https://github.com/tdrussell/diffusion-pipe) by tdrussell. We thank the Wan2.1 team for releasing the backbone model.
