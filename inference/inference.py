#!/usr/bin/env python3
"""
Unified inference script for Less Is More (LIM) — FPS conditioning for Wan2.1 T2V.

Prompt input modes (mutually exclusive):
  --prompt TEXT         Single inline prompt string.
  --prompt_file FILE    One prompt per line in a .txt file.
  --prompt_dir DIR      Directory of .txt category files; each line = one prompt.
                        Outputs are saved in per-category subdirectories.
  --prompt_folder DIR   Directory of .txt files; each file = one prompt (whole
                        file content is the prompt text).

Inference modes (mutually exclusive; default is "dirty" — full LoRA + FPS):
  --graft               Paper's clean method: LoRA all blocks, FPS adapter in
                        deepest-third blocks only.
  --clean               Pure backbone; no LoRA, no FPS adapters loaded.
  --fps_only            Load only FPS-related parameters from the checkpoint.
  --base_only           Load only base LoRA parameters from the checkpoint.
  (no flag)             Normal/dirty mode: load all parameters.

Alignment:
  --align               Enable magnitude alignment for --fps_only mode.
                        Calibrates rboth_ijk = ||y_fps|| / ||y_text|| per block,
                        then scales y_fps during inference to match the training
                        distribution. Requires a checkpoint trained WITH base LoRA.

Checkpoint sweep:
  --checkpoint_parent DIR --checkpoint_prefix PREFIX
  --checkpoint_start N --checkpoint_end N --checkpoint_interval N
  Discovers checkpoint subfolders matching PREFIX<N> in DIR and runs inference
  for each, outputting results in per-checkpoint subdirectories.

Output naming:
  {line_id:05d}_{first_5_words_of_prompt}_{fps_value}.mp4
  For modes with no FPS conditioning (--clean, --base_only):
      {line_id:05d}_{first_5_words_of_prompt}.mp4
"""

import torch
import os
import sys
import re
import logging
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
import deepspeed
from tqdm import tqdm
import math
import argparse
import toml
import json

from models.wan.wan import WanPipeline
import peft
from inference_utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from inference_utils.utils import cache_video
from utils.common import DTYPE_MAP


# ---------------------------------------------------------------------------
# Environment / pipeline setup
# ---------------------------------------------------------------------------

def setup_environment(port='29501'):
    """Initializes the DeepSpeed distributed environment for standalone use."""
    if not deepspeed.comm.is_initialized():
        os.environ.setdefault('MASTER_ADDR', 'localhost')
        os.environ.setdefault('MASTER_PORT', port)
        os.environ.setdefault('RANK', '0')
        os.environ.setdefault('WORLD_SIZE', '1')
        os.environ.setdefault('LOCAL_RANK', '0')
        deepspeed.init_distributed()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
    logging.info(f"DeepSpeed environment initialized on port {port}.")


def load_pipeline_from_toml(toml_path):
    """Loads WanPipeline using TOML configuration (training-aligned)."""
    logging.info(f"Loading TOML configuration: {toml_path}")

    with open(toml_path) as f:
        config = json.loads(json.dumps(toml.load(f)))

    model_dtype_str = config['model']['dtype']
    config['model']['dtype'] = DTYPE_MAP[model_dtype_str]
    if transformer_dtype := config['model'].get('transformer_dtype', None):
        config['model']['transformer_dtype'] = DTYPE_MAP.get(transformer_dtype, transformer_dtype)

    logging.info("Initializing WanPipeline with TOML configuration...")
    logging.info(f"Model type: {config['model']['type']}")
    logging.info(f"Model checkpoint: {config['model']['ckpt_path']}")
    logging.info(f"Model dtype: {model_dtype_str} -> {config['model']['dtype']}")

    model_config = config['model']
    fps_settings = {
        'fps_adapter_rank': model_config.get('fps_adapter_rank'),
        'fps_adapter_gate_init': model_config.get('fps_adapter_gate_init'),
        'fps_condition_blocks': model_config.get('fps_condition_blocks'),
        'fps_adapter_num_tokens': model_config.get('fps_adapter_num_tokens'),
        'fps_tau_transform': model_config.get('fps_tau_transform'),
        'fps_tau_scale': model_config.get('fps_tau_scale'),
        'fps_embed_dim': model_config.get('fps_embed_dim'),
        'fps_condition_hidden': model_config.get('fps_condition_hidden'),
        'fps_lora_alpha': model_config.get('fps_lora_alpha'),
        'fps_gate_mode': model_config.get('fps_gate_mode'),
        'fps_gate_fixed_value': model_config.get('fps_gate_fixed_value'),
        'fps_scale': model_config.get('fps_scale'),
        'fps_warmup_steps': model_config.get('fps_warmup_steps'),
    }
    logging.info("Complete FPS configuration from TOML (aligned with wan.py):")
    for key, value in fps_settings.items():
        if value is not None:
            logging.info(f"  {key}: {value}")
        else:
            logging.info(f"  {key}: <not set, will use wan.py default>")

    wan_t2v = WanPipeline(config)

    logging.info("Loading main transformer model...")
    wan_t2v.load_diffusion_model()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Moving model components to device: {device}")

    wan_t2v.transformer.to(device)
    wan_t2v.vae.model.to(device)
    wan_t2v.text_encoder.model.to(device)

    logging.info("Setting models to eval mode for inference...")
    wan_t2v.transformer.eval()
    wan_t2v.vae.model.eval()
    wan_t2v.text_encoder.model.eval()

    logging.info("Pipeline loaded successfully with TOML configuration.")
    return wan_t2v, config


# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------

def detect_checkpoint_type(lora_path):
    """Detects FPS-only vs mixed checkpoint type."""
    import safetensors
    try:
        checkpoint = safetensors.torch.load_file(lora_path)

        has_base_lora = False
        has_fps_params = False

        for key in checkpoint.keys():
            if 'fps_conditioning' in key or 'fps_adapter' in key:
                has_fps_params = True
            elif key.endswith('.weight') and ('lora_A' in key or 'lora_B' in key):
                if 'fps' not in key:
                    has_base_lora = True

        if has_fps_params and not has_base_lora:
            return 'fps_only'
        elif has_fps_params and has_base_lora:
            return 'lora_and_fps'
        else:
            return 'unknown'
    except Exception as e:
        logging.warning(f"Could not detect checkpoint type: {e}")
        return 'unknown'


def get_fps_block_indices(config):
    """Extract the list of block indices where FPS adapters are injected.

    Returns:
        List of block indices (e.g., [27, 28, 29, ..., 38, 39])
    """
    model_config = config.get('model', {})
    fps_condition_blocks = model_config.get('fps_condition_blocks', 'default')

    if fps_condition_blocks in ['default', 'deepest_third']:
        return list(range(27, 40))
    elif isinstance(fps_condition_blocks, str) and '-' in fps_condition_blocks:
        start, end = map(int, fps_condition_blocks.split('-'))
        return list(range(start, end + 1))
    elif isinstance(fps_condition_blocks, list):
        return fps_condition_blocks
    else:
        try:
            return json.loads(fps_condition_blocks)
        except Exception:
            logging.warning(f"Could not parse fps_condition_blocks: {fps_condition_blocks}, using default [27-39]")
            return list(range(27, 40))


def load_adapter_weights_selective(pipeline, checkpoint_path, fps_only=False, base_only=False,
                                   graft_mode=False, fps_block_indices=None):
    """Load adapter weights with selective parameter filtering using temporary files."""
    if not fps_only and not base_only and not graft_mode:
        logging.info("Loading all parameters (normal mode)")
        pipeline.load_adapter_weights(checkpoint_path)
        return

    import safetensors

    checkpoint_dir = Path(checkpoint_path)
    if checkpoint_dir.is_dir():
        safetensors_files = list(checkpoint_dir.glob('*.safetensors'))
        if not safetensors_files:
            raise FileNotFoundError(f"No .safetensors files found in {checkpoint_dir}")
        checkpoint_file = safetensors_files[0]
        logging.info(f"Found checkpoint file: {checkpoint_file}")
    else:
        checkpoint_file = checkpoint_dir

    logging.info(f"Loading checkpoint: {checkpoint_file}")
    checkpoint = safetensors.torch.load_file(str(checkpoint_file))

    if fps_only:
        filtered_checkpoint = {k: v for k, v in checkpoint.items()
                               if 'fps_conditioning' in k or 'fps_adapter' in k}
        mode_name = "FPS-ONLY"
        logging.info(f"FPS-ONLY MODE: Filtered {len(filtered_checkpoint)} FPS parameters "
                     f"out of {len(checkpoint)} total")

    elif base_only:
        filtered_checkpoint = {k: v for k, v in checkpoint.items()
                               if 'fps_conditioning' not in k and 'fps_adapter' not in k}
        mode_name = "BASE-ONLY"
        logging.info(f"BASE-ONLY MODE: Filtered {len(filtered_checkpoint)} base LoRA parameters "
                     f"out of {len(checkpoint)} total")

    elif graft_mode:
        if fps_block_indices is None:
            raise ValueError("GRAFT mode requires fps_block_indices to be specified")

        logging.info(f"GRAFT MODE: Loading FPS adapters + base LoRA only in blocks {fps_block_indices}")

        filtered_checkpoint = {}
        for k, v in checkpoint.items():
            if 'fps_conditioning' in k:
                filtered_checkpoint[k] = v
            elif 'fps_adapter' in k:
                filtered_checkpoint[k] = v
            else:
                if '.blocks.' in k:
                    try:
                        parts = k.split('.blocks.')[1].split('.')
                        block_idx = int(parts[0])
                        if block_idx in fps_block_indices:
                            filtered_checkpoint[k] = v
                    except (IndexError, ValueError):
                        pass

        mode_name = "GRAFT"
        logging.info(f"GRAFT MODE: Filtered {len(filtered_checkpoint)} parameters "
                     f"(FPS + grafted base LoRA) out of {len(checkpoint)} total")

    fps_conditioning_params = [k for k in checkpoint.keys() if 'fps_conditioning' in k]
    fps_adapter_params = [k for k in checkpoint.keys() if 'fps_adapter' in k]
    base_lora_params = [k for k in checkpoint.keys()
                        if 'fps_conditioning' not in k and 'fps_adapter' not in k]

    logging.info(f"Parameter breakdown in original checkpoint:")
    logging.info(f"  FPS conditioning (MLP): {len(fps_conditioning_params)} parameters")
    logging.info(f"  FPS adapters (LoRA): {len(fps_adapter_params)} parameters")
    logging.info(f"  Base LoRA: {len(base_lora_params)} parameters")
    logging.info(f"  Total: {len(checkpoint)} parameters")

    if fps_only:
        logging.info(f"FPS-ONLY filtering results:")
        logging.info(f"  Loading FPS conditioning: {len(fps_conditioning_params)} parameters")
        logging.info(f"  Loading FPS adapters: {len(fps_adapter_params)} parameters")
        logging.info(f"  Skipping base LoRA: {len(base_lora_params)} parameters")

    elif base_only:
        logging.info(f"BASE-ONLY filtering results:")
        logging.info(f"  Loading base LoRA: {len(base_lora_params)} parameters")
        logging.info(f"  Skipping FPS conditioning: {len(fps_conditioning_params)} parameters")
        logging.info(f"  Skipping FPS adapters: {len(fps_adapter_params)} parameters")

    elif graft_mode:
        grafted_base_lora = [k for k in filtered_checkpoint.keys()
                             if 'fps_conditioning' not in k and 'fps_adapter' not in k]
        all_blocks_with_base_lora = set()
        for param_name in grafted_base_lora:
            if '.blocks.' in param_name:
                parts = param_name.split('.blocks.')[1].split('.')
                block_idx = int(parts[0])
                all_blocks_with_base_lora.add(block_idx)

        logging.info(f"GRAFT filtering results:")
        logging.info(f"  Loading FPS conditioning: {len(fps_conditioning_params)} parameters")
        logging.info(f"  Loading FPS adapters: {len(fps_adapter_params)} parameters")
        logging.info(f"  Loading GRAFTED base LoRA "
                     f"(blocks {sorted(all_blocks_with_base_lora)}): {len(grafted_base_lora)} parameters")
        logging.info(f"  Skipping base LoRA from other blocks: "
                     f"{len(base_lora_params) - len(grafted_base_lora)} parameters")

    temp_dir = tempfile.mkdtemp(prefix=f"selective_loading_{mode_name.lower()}_")
    temp_checkpoint_file = Path(temp_dir) / "filtered_adapter.safetensors"

    logging.info(f"Creating temporary filtered checkpoint: {temp_checkpoint_file}")
    safetensors.torch.save_file(filtered_checkpoint, str(temp_checkpoint_file))

    try:
        pipeline.load_adapter_weights(temp_dir)
        logging.info(f"Successfully loaded {len(filtered_checkpoint)} filtered parameters")
    finally:
        shutil.rmtree(temp_dir)
        logging.info(f"Cleaned up temporary directory: {temp_dir}")


def verify_fps_config_applied(pipeline, config):
    """Verify that FPS configuration from TOML was properly applied to the model."""
    model_config = config['model']
    logging.info("Verifying FPS configuration from TOML was applied correctly:")

    fps_adapter_count = sum(1 for n, _ in pipeline.transformer.named_parameters() if 'fps_adapter' in n)
    fps_conditioning_count = sum(1 for n, _ in pipeline.transformer.named_parameters() if 'fps_conditioning' in n)

    logging.info(f"  Found {fps_adapter_count} FPS adapter parameters in model")
    logging.info(f"  Found {fps_conditioning_count} FPS conditioning (MLP) parameters in model")

    expected_settings = {
        'fps_adapter_rank': model_config.get('fps_adapter_rank', 'default'),
        'fps_condition_blocks': model_config.get('fps_condition_blocks', 'default'),
        'fps_gate_mode': model_config.get('fps_gate_mode', 'default'),
        'fps_gate_fixed_value': model_config.get('fps_gate_fixed_value', 'default'),
        'fps_adapter_num_tokens': model_config.get('fps_adapter_num_tokens', 'default'),
        'fps_embed_dim': model_config.get('fps_embed_dim', 'default'),
        'fps_reference_fps': model_config.get('fps_reference_fps', 'default (240.0)'),
        'fps_scale': model_config.get('fps_scale', 'default (False)'),
        'fps_warmup_steps': model_config.get('fps_warmup_steps', 'default (100)'),
    }

    logging.info("  Expected FPS configuration from TOML:")
    for key, value in expected_settings.items():
        logging.info(f"    {key}: {value}")

    has_gate_alpha = any('gate_alpha' in n for n, _ in pipeline.transformer.named_parameters())
    has_gate_fixed = any('gate_fixed' in n for n, _ in pipeline.transformer.named_buffers())

    if has_gate_alpha and not has_gate_fixed:
        detected_mode = "learnable (sigmoid/identity/relu/silu/softplus)"
    elif has_gate_fixed and not has_gate_alpha:
        detected_mode = "fixed"
    elif has_gate_alpha and has_gate_fixed:
        detected_mode = "mixed (both found)"
    else:
        detected_mode = "unknown"

    expected_mode = model_config.get('fps_gate_mode', 'sigmoid')
    learnable_modes = ['sigmoid', 'identity', 'relu', 'silu', 'softplus']
    expected_is_learnable = expected_mode in learnable_modes
    detected_is_learnable = has_gate_alpha and not has_gate_fixed
    mode_match = ("OK" if (expected_is_learnable == detected_is_learnable)
                  or (expected_mode == 'fixed' and detected_mode == 'fixed') else "MISMATCH")
    logging.info(f"  Gate mode check: Expected='{expected_mode}', Detected='{detected_mode}' [{mode_match}]")

    actual_reference_fps = getattr(pipeline, 'fps_reference_fps', 'not found')
    expected_reference_fps = model_config.get('fps_reference_fps', 240.0)
    reference_match = "OK" if actual_reference_fps == expected_reference_fps else "MISMATCH"
    logging.info(f"  Reference FPS check: Expected={expected_reference_fps}, "
                 f"Actual={actual_reference_fps} [{reference_match}]")


def apply_checkpoint(wan_t2v_pipeline, checkpoint_path, rank=32, dtype=torch.bfloat16,
                     fps_only=False, base_only=False, graft_mode=False, config=None):
    """Applies checkpoint weights (FPS-only, base-only, GRAFT, or mixed LoRA+FPS)."""
    logging.info(f"Loading checkpoint: {checkpoint_path}")

    checkpoint_type = detect_checkpoint_type(checkpoint_path)
    logging.info(f"Detected checkpoint type: {checkpoint_type}")

    if (fps_only or graft_mode) and config:
        verify_fps_config_applied(wan_t2v_pipeline, config)

    configure_base_lora = False

    if fps_only:
        logging.info("FPS-ONLY MODE: Skipping base LoRA configuration")
        configure_base_lora = False
    elif base_only:
        logging.info("BASE-ONLY MODE: Configuring only base LoRA adapter")
        configure_base_lora = True
    elif graft_mode:
        logging.info("GRAFT MODE: Configuring base LoRA adapter (will be loaded selectively)")
        configure_base_lora = True
    else:
        if checkpoint_type in ['lora_and_fps', 'unknown']:
            configure_base_lora = True
        elif checkpoint_type == 'fps_only':
            configure_base_lora = False

    if configure_base_lora:
        adapter_config = {"type": "lora", "rank": rank, "alpha": rank, "dropout": 0.0, "dtype": dtype}
        wan_t2v_pipeline.configure_adapter(adapter_config)
        logging.info(f"Configured base LoRA adapter with rank {rank}")

    fps_block_indices = None
    if graft_mode:
        if config is None:
            raise ValueError("GRAFT mode requires config to determine FPS block indices")
        fps_block_indices = get_fps_block_indices(config)
        logging.info(f"GRAFT MODE: Extracted FPS block indices from config: {fps_block_indices}")

    load_adapter_weights_selective(wan_t2v_pipeline, checkpoint_path,
                                   fps_only, base_only, graft_mode, fps_block_indices)

    fps_mlp_params = 0
    fps_adapter_params_count = 0
    base_lora_params_count = 0

    for name, param in wan_t2v_pipeline.transformer.named_parameters():
        if 'fps_conditioning' in name:
            fps_mlp_params += 1
        elif 'fps_adapter' in name:
            fps_adapter_params_count += 1
        elif 'lora' in name.lower() and 'fps' not in name:
            base_lora_params_count += 1

    logging.info(f"Checkpoint loaded successfully!")
    logging.info(f"  FPS MLP parameters: {fps_mlp_params}")
    logging.info(f"  FPS Adapter parameters: {fps_adapter_params_count}")
    logging.info(f"  Base LoRA parameters: {base_lora_params_count}")

    if fps_mlp_params == 0 and fps_adapter_params_count == 0:
        logging.warning("No FPS parameters found!")

    return wan_t2v_pipeline


def validate_fps_config(config):
    """Validates that the config has FPS parameter definitions for selective loading."""
    model_config = config.get('model', {})

    fps_params = [
        'fps_adapter_rank', 'fps_adapter_num_tokens', 'fps_embed_dim',
        'fps_lora_alpha', 'fps_condition_blocks',
    ]

    missing_params = [param for param in fps_params if param not in model_config]
    if missing_params:
        raise ValueError(
            f"--fps_only requires FPS parameters in config, but missing: {missing_params}. "
            f"Cannot load FPS-only parameters from checkpoint."
        )

    logging.info("Config validation passed: FPS parameters found in configuration")


def validate_base_lora_config(config):
    """Validates that the config has base LoRA parameter definitions for selective loading."""
    if 'adapter' not in config:
        raise ValueError(
            "--base_only requires [adapter] section in config, but it's missing. "
            "Cannot load base LoRA parameters from checkpoint."
        )

    adapter_config = config['adapter']
    required_params = ['type', 'rank']
    missing_params = [param for param in required_params if param not in adapter_config]

    if missing_params:
        raise ValueError(
            f"--base_only requires adapter parameters in config, but missing: {missing_params}. "
            f"Cannot load base LoRA parameters from checkpoint."
        )

    if adapter_config['type'] != 'lora':
        raise ValueError(
            f"--base_only requires adapter type 'lora', but found '{adapter_config['type']}'. "
            f"Cannot load base LoRA parameters from checkpoint."
        )

    logging.info("Config validation passed: Base LoRA parameters found in configuration")


def discover_checkpoints(parent_folder, prefix, start, end, interval):
    """
    Discover checkpoint subfolders based on prefix and range.

    Returns:
        List of (checkpoint_name, checkpoint_path) tuples
    """
    parent_path = Path(parent_folder)
    if not parent_path.exists() or not parent_path.is_dir():
        raise ValueError(f"Checkpoint parent folder does not exist or is not a directory: {parent_folder}")

    checkpoints = []
    for checkpoint_num in range(start, end + 1, interval):
        checkpoint_name = f"{prefix}{checkpoint_num}"
        checkpoint_path = parent_path / checkpoint_name

        if checkpoint_path.exists() and checkpoint_path.is_dir():
            safetensors_files = list(checkpoint_path.glob('*.safetensors'))
            if safetensors_files:
                checkpoints.append((checkpoint_name, str(checkpoint_path)))
                logging.info(f"  Found checkpoint: {checkpoint_name}")
            else:
                logging.warning(f"  Skipping {checkpoint_name}: no .safetensors files found")
        else:
            logging.warning(f"  Skipping {checkpoint_name}: folder does not exist")

    if not checkpoints:
        raise ValueError(
            f"No valid checkpoints found in {parent_folder} with prefix '{prefix}' "
            f"and range [{start}, {end}] interval {interval}"
        )

    return checkpoints


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

def load_prompts_from_file(file_path):
    """
    Load prompts from a single text file where each line is a prompt.

    Returns:
        List of (line_id, prompt_text, category) tuples where category is None.
        line_id is 0-based index (integer), formatted later.
    """
    file_path = Path(file_path)
    if not file_path.exists() or not file_path.is_file():
        raise ValueError(f"Prompt file does not exist or is not a file: {file_path}")

    prompts = []
    logging.info(f"Loading prompts from file: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        line_idx = 0
        for line in f:
            prompt_text = line.strip()
            if prompt_text:
                prompts.append((line_idx, prompt_text, None))
                logging.info(f"  Prompt {line_idx:05d}: {prompt_text[:60]}...")
                line_idx += 1

    if not prompts:
        raise ValueError(f"No valid prompts loaded from file: {file_path}")

    logging.info(f"Total: {len(prompts)} prompts loaded")
    return prompts


def load_prompts_from_directory(dir_path):
    """
    Load prompts from multiple text files in a directory.
    Each txt file contains prompts (one per line).

    Returns:
        List of (line_id, prompt_text, category) tuples.
        line_id is a per-category 0-based index (integer).
        category is the txt file stem (e.g. 'animal' from 'animal.txt').
    """
    dir_path = Path(dir_path)
    if not dir_path.exists() or not dir_path.is_dir():
        raise ValueError(f"Prompt directory does not exist or is not a directory: {dir_path}")

    txt_files = sorted(dir_path.glob('*.txt'))
    if not txt_files:
        raise ValueError(f"No .txt files found in directory: {dir_path}")

    prompts = []
    logging.info(f"Loading prompts from directory: {dir_path}")
    logging.info(f"  Found {len(txt_files)} category files")

    for txt_file in txt_files:
        category = txt_file.stem
        logging.info(f"  Category: {category}")

        line_idx = 0
        with open(txt_file, 'r', encoding='utf-8') as f:
            for line in f:
                prompt_text = line.strip()
                if prompt_text:
                    prompts.append((line_idx, prompt_text, category))
                    if line_idx < 3:
                        logging.info(f"    {line_idx:05d}: {prompt_text[:60]}...")
                    line_idx += 1

        cat_count = sum(1 for _, _, c in prompts if c == category)
        logging.info(f"    Loaded {cat_count} prompts from {category}")

    if not prompts:
        raise ValueError(f"No valid prompts loaded from directory: {dir_path}")

    logging.info(f"Total: {len(prompts)} prompts loaded from {len(txt_files)} categories")
    return prompts


def load_prompts_from_inline(prompt_text):
    """
    Wrap a single inline prompt string as a prompt list.

    Returns:
        List with one (line_id, prompt_text, category) tuple.
    """
    return [(0, prompt_text.strip(), None)]


def load_prompts_from_folder(folder_path):
    """
    Load prompts from text files in a folder where each file is one prompt.
    The whole file content (stripped) becomes the prompt text.

    Returns:
        List of (line_id, prompt_text, category) tuples.
        line_id is the 0-based sorted file index.
        category is None (no subdirectory nesting).
    """
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"Prompt folder does not exist or is not a directory: {folder_path}")

    txt_files = sorted(folder.glob('*.txt'))
    if not txt_files:
        raise ValueError(f"No .txt files found in prompt folder: {folder_path}")

    prompts = []
    logging.info(f"Loading prompts from folder: {folder_path}")

    for idx, txt_file in enumerate(txt_files):
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                prompt_text = f.read().strip()
                if prompt_text:
                    prompts.append((idx, prompt_text, None))
                    logging.info(f"  Loaded '{txt_file.stem}': {prompt_text[:60]}...")
        except Exception as e:
            logging.warning(f"  Failed to read {txt_file.name}: {e}")

    if not prompts:
        raise ValueError(f"No valid prompts loaded from folder: {folder_path}")

    logging.info(f"Total: {len(prompts)} prompts loaded")
    return prompts


# ---------------------------------------------------------------------------
# FPS value parsing
# ---------------------------------------------------------------------------

def parse_condition_values(condition_values_flat, num_conditions):
    """
    Parse flat condition_values list into groups based on num_conditions.

    Single-condition (num_conditions=1):
        Input:  [-1.0, 0.0, 1.0]
        Output: [-1.0, 0.0, 1.0]

    Multi-condition (num_conditions=2):
        Input:  [0.5, 0.02, 1.0, 0.05, 0.125, 0.1]
        Output: [[0.5, 0.02], [1.0, 0.05], [0.125, 0.1]]
    """
    if num_conditions == 1:
        return condition_values_flat
    else:
        if len(condition_values_flat) % num_conditions != 0:
            raise ValueError(
                f"condition_values length ({len(condition_values_flat)}) must be divisible by "
                f"num_conditions ({num_conditions}). Got {len(condition_values_flat)} values "
                f"which doesn't divide evenly into groups of {num_conditions}."
            )
        grouped = []
        for i in range(0, len(condition_values_flat), num_conditions):
            grouped.append(condition_values_flat[i:i + num_conditions])
        return grouped


# ---------------------------------------------------------------------------
# Output filename helpers
# ---------------------------------------------------------------------------

def _first_5_words_slug(prompt_text):
    """Return a slug of the first 5 words of a prompt for use in a filename."""
    words = prompt_text.split()[:5]
    joined = ' '.join(words).lower()
    slug = re.sub(r'[^a-z0-9]+', '_', joined)
    slug = slug.strip('_')
    slug = re.sub(r'_+', '_', slug)
    return slug


def build_output_filename(line_id, prompt_text, fps=None):
    """
    Build the canonical output filename.

    Args:
        line_id: 0-based integer prompt index.
        prompt_text: Full prompt string.
        fps: FPS scalar value (float, int, or list). Pass None for
             modes with no FPS conditioning (--clean, --base_only).

    Returns:
        Filename string, e.g. '00001_a_cheetah_in_full_sprint_0.5.mp4'
    """
    slug = _first_5_words_slug(prompt_text)
    prefix = f"{line_id:05d}_{slug}"

    if fps is None:
        return f"{prefix}.mp4"

    if isinstance(fps, list):
        fps_str = '_'.join(str(v) for v in fps)
    else:
        fps_str = str(fps)

    return f"{prefix}_{fps_str}.mp4"


# ---------------------------------------------------------------------------
# Alignment calibration
# ---------------------------------------------------------------------------

def calibrate_alignment_ratios(pipeline, config, prompt, n_prompt, fps_values,
                                seed, steps, scale, frames, size, shift):
    """
    Measure rboth_ijk = ||y_fps|| / ||y_text|| with base LoRA + FPS.

    Returns dict: {(step_i, block_j, fps_k): rboth_ijk}

    This ratio is used during fps_only inference to scale y_fps to match
    the training distribution:
        y_fps_aligned = y_fps * (rboth_ijk / rfps_ijk)

    Steps:
    1. Run full inference with base LoRA + FPS for all FPS values.
    2. At each denoising step, capture ||y_fps|| and ||y_text|| per block.
    3. Store rboth_ijk = ||y_fps|| / ||y_text|| for all (i, j, k).
    """
    logging.info("Running alignment calibration pass...")
    logging.info("  Measuring rboth_ijk = ||y_fps|| / ||y_text|| at each denoising step")

    rboth_ratios = {}

    fps_adapter_block_indices = []
    for block_idx, block in enumerate(pipeline.transformer.blocks):
        if hasattr(block, 'fps_adapter') and block.fps_adapter is not None:
            fps_adapter_block_indices.append(block_idx)

    logging.info(f"  Found {len(fps_adapter_block_indices)} blocks with FPS adapters")

    base_lora_count = sum(1 for n, _ in pipeline.transformer.named_parameters()
                         if '.lora_' in n and 'fps' not in n)
    logging.info(f"  Found {base_lora_count} base LoRA parameters in model")

    if base_lora_count == 0:
        logging.warning("  NO BASE LoRA DETECTED!")
        logging.warning("  Alignment requires a checkpoint trained WITH base LoRA.")
        logging.warning("  Skipping calibration - returning empty ratios.")
        return {}

    print(f"\nCALIBRATION: Running inference with base LoRA + FPS for "
          f"{len(fps_values)} condition values", flush=True)
    print(f"  Will capture rboth_ijk = ||y_fps|| / ||y_text|| at ALL denoising steps",
          flush=True)

    for fps_idx, fps_value in enumerate(fps_values):
        print(f"\n  Calibrating condition={fps_value} ({fps_idx+1}/{len(fps_values)})...", flush=True)

        _ = generate_video_with_fps(
            pipeline, config, prompt, n_prompt, fps_value, seed, steps, scale,
            frames, size, shift,
            force_gate_one=False,
            alignment_ratios=None,
            capture_rboth=True,
            rboth_storage=rboth_ratios,
            fps_idx=fps_idx,
        )

    print(f"\nCalibration complete: Captured {len(rboth_ratios)} rboth_ijk ratios", flush=True)
    print(f"  Total: {len(fps_values)} condition values x {steps} steps x "
          f"{len(fps_adapter_block_indices)} blocks", flush=True)

    return rboth_ratios


# ---------------------------------------------------------------------------
# Core video generation
# ---------------------------------------------------------------------------

def generate_video_with_fps(pipeline, config, prompt, n_prompt, fps=60, seed=42, steps=25,
                             scale=7.0, frames=49, size=(512, 320), shift=3.0,
                             force_gate_one=False, alignment_ratios=None,
                             capture_rboth=False, rboth_storage=None, fps_idx=None):
    """Generates a video with specific FPS conditioning.

    Args:
        pipeline: WanPipeline instance.
        config: TOML configuration dict.
        prompt: Text prompt.
        n_prompt: Negative prompt.
        fps: FPS conditioning value (scalar or list for multi-condition).
        seed: Random seed.
        steps: Number of denoising steps.
        scale: CFG guidance scale.
        frames: Number of video frames.
        size: (width, height) tuple.
        shift: Scheduler shift parameter.
        force_gate_one: Force all FPS adapter gates to 1.0.
        alignment_ratios: Dict of per-block alignment ratios for --align mode.
        capture_rboth: If True, capture rboth_ijk ratios during generation.
        rboth_storage: Dict to store captured ratios.
        fps_idx: FPS index k for storage key.
    """
    logging.info(f"Generating video with FPS={fps}")
    logging.info(f"  Prompt: {prompt}")
    logging.info(f"  Steps: {steps}, Frames: {frames}, Size: {size}")

    device = pipeline.transformer.device

    torch.manual_seed(seed)

    text_encoder_fn = pipeline.get_call_text_encoder_fn(pipeline.text_encoder.model.to(device))
    cond_inputs = text_encoder_fn([prompt], is_video=True)
    uncond_inputs = text_encoder_fn([n_prompt], is_video=True)

    fps_values = torch.tensor([fps], dtype=torch.float32, device=device)
    logging.info(f"  FPS tensor: {fps_values.tolist()}")

    if force_gate_one:
        logging.info("DIAGNOSTIC MODE: Forcing all FPS adapter gates to 1.0")
        gates_forced = 0

        for name, param in pipeline.transformer.named_parameters():
            if 'fps_adapter' in name and 'gate_alpha' in name:
                with torch.no_grad():
                    param.data.fill_(10.0)
                gates_forced += 1

        for name, buffer in pipeline.transformer.named_buffers():
            if 'fps_adapter' in name and 'gate_fixed' in name:
                with torch.no_grad():
                    buffer.data.fill_(1.0)
                gates_forced += 1

        logging.info(f"  Forced {gates_forced} gate values to maximum (1.0)")

    alignment_hooks = []
    if alignment_ratios is not None:
        logging.info("Applying magnitude alignment for fps_only mode")

        def make_scaling_forward(original_forward, block_idx, scale_ratio):
            """Wraps FPS adapter forward to scale y_fps by alignment ratio."""
            def scaling_forward(q, k_text, v_text, fps_conditioning, context_lens):
                from models.wan.attention import flash_attention

                y_text = flash_attention(q, k_text, v_text, k_lens=context_lens)

                module = scaling_forward.__self__

                k_fps_proj = module.k_fps_up(module.k_fps_down(fps_conditioning))
                v_fps_proj = module.v_fps_up(module.v_fps_down(fps_conditioning))

                B = q.size(0)
                k_fps_proj = k_fps_proj.view(B * module.num_tokens, module.dim)
                k_fps_proj = module.norm_k_fps(k_fps_proj)
                k_fps_proj = k_fps_proj.view(B, module.num_tokens * module.dim)

                k_fps_proj = k_fps_proj * module.lora_scale
                v_fps_proj = v_fps_proj * module.lora_scale

                k_fps = k_fps_proj.view(B, module.num_tokens, module.num_heads, module.head_dim)
                v_fps = v_fps_proj.view(B, module.num_tokens, module.num_heads, module.head_dim)

                y_fps = flash_attention(q, k_fps, v_fps, k_lens=None)

                if module.fps_scale:
                    y_text_norm = torch.norm(y_text)
                    y_fps_norm = torch.norm(y_fps)
                    y_fps_norm_safe = y_fps_norm + 1e-8
                    scale_factor = y_text_norm / y_fps_norm_safe
                    y_fps = y_fps * scale_factor

                if module.gate_mode == 'sigmoid':
                    gate = torch.sigmoid(module.gate_alpha)
                elif module.gate_mode == 'identity':
                    gate = module.gate_alpha
                elif module.gate_mode == 'relu':
                    gate = torch.relu(module.gate_alpha)
                elif module.gate_mode == 'silu':
                    gate = torch.nn.functional.silu(module.gate_alpha)
                elif module.gate_mode == 'softplus':
                    gate = torch.nn.functional.softplus(module.gate_alpha)
                elif module.gate_mode == 'fixed':
                    gate = module.gate_fixed
                else:
                    gate = 1.0

                warmup_factor = 1.0

                y_fps_scaled = y_fps * scale_ratio

                y_combined = y_text + (warmup_factor * gate) * y_fps_scaled
                return y_combined

            return scaling_forward

        for block_idx, block in enumerate(pipeline.transformer.blocks):
            if hasattr(block, 'fps_adapter') and block.fps_adapter is not None:
                if block_idx in alignment_ratios:
                    ratio = alignment_ratios[block_idx]
                    logging.info(f"  Block {block_idx}: Applying scale ratio {ratio:.4f}")

                    original_forward = block.fps_adapter.forward
                    scaling_forward = make_scaling_forward(original_forward, block_idx, ratio)
                    scaling_forward.__self__ = block.fps_adapter
                    block.fps_adapter.forward = scaling_forward
                    alignment_hooks.append((block.fps_adapter, original_forward))

        logging.info(f"  Applied alignment to {len(alignment_hooks)} FPS adapter blocks")

    vae_stride = [4, 8, 8]
    target_shape = (16, frames // vae_stride[0], size[1] // vae_stride[1], size[0] // vae_stride[2])
    latents = torch.randn(target_shape, device=device)

    layers = pipeline.to_layers()
    initial_layer, transformer_layers, final_layer = layers[0], layers[1:-1], layers[-1]

    scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
    scheduler.set_timesteps(steps, device=device, shift=shift)
    timesteps = scheduler.timesteps

    for step_idx, t in enumerate(tqdm(timesteps, desc=f"FPS={fps}")):
        t_batch = torch.full((1,), t, device=device)

        autocast_dtype = config['model']['dtype']

        with torch.no_grad(), torch.autocast('cuda', dtype=autocast_dtype):
            initial_input_uncond = (
                latents.unsqueeze(0), None, t_batch,
                uncond_inputs['text_embeddings'].to(device),
                uncond_inputs['seq_lens'].to(device),
                None, fps_values
            )

            layer_outputs = initial_layer(initial_input_uncond)
            x, e, e0, seq_lens, grid_sizes, freqs, context, fps_conditioning = layer_outputs

            for transformer_layer in transformer_layers:
                layer_outputs = transformer_layer((x, e, e0, seq_lens, grid_sizes, freqs, context, fps_conditioning))
                x, e, e0, seq_lens, grid_sizes, freqs, context, fps_conditioning = layer_outputs

            noise_pred_uncond = final_layer((x, e, e0, seq_lens, grid_sizes, freqs, context, fps_conditioning))

            initial_input_cond = (
                latents.unsqueeze(0), None, t_batch,
                cond_inputs['text_embeddings'].to(device),
                cond_inputs['seq_lens'].to(device),
                None, fps_values
            )

            layer_outputs = initial_layer(initial_input_cond)
            x, e, e0, seq_lens, grid_sizes, freqs, context, fps_conditioning = layer_outputs

            for transformer_layer in transformer_layers:
                layer_outputs = transformer_layer((x, e, e0, seq_lens, grid_sizes, freqs, context, fps_conditioning))
                x, e, e0, seq_lens, grid_sizes, freqs, context, fps_conditioning = layer_outputs

            noise_pred_cond = final_layer((x, e, e0, seq_lens, grid_sizes, freqs, context, fps_conditioning))

            noise_pred = (noise_pred_uncond.squeeze(0)
                         + scale * (noise_pred_cond.squeeze(0) - noise_pred_uncond.squeeze(0)))

        latents = scheduler.step(noise_pred, t, latents.unsqueeze(0), return_dict=False)[0].squeeze(0)

    logging.info("Decoding latents...")
    video_tensor = pipeline.vae.decode([latents])[0]

    logging.info(f"Generation complete for FPS={fps}")

    return video_tensor


# ---------------------------------------------------------------------------
# Batch inference runner
# ---------------------------------------------------------------------------

def run_batch_inference(pipeline, config, prompts, fps_values, output_dir,
                        include_fps_in_name, use_alignment=False,
                        force_gate_one=False, **generation_kwargs):
    """
    Runs batch inference for all prompts and FPS values.

    Args:
        pipeline: WanPipeline instance.
        config: TOML configuration dict.
        prompts: List of (line_id, prompt_text, category) tuples.
                 line_id is a 0-based integer.
        fps_values: List of FPS values (parsed for num_conditions).
        output_dir: Base output directory.
        include_fps_in_name: Whether to include FPS value in filename.
        use_alignment: Run alignment calibration before each prompt's inference.
        force_gate_one: Force FPS adapter gates to 1.0.
        **generation_kwargs: seed, steps, scale, frames, size, shift, n_prompt.

    Returns:
        List of per-prompt result dicts.
    """
    logging.info("Starting batch inference...")
    logging.info(f"  Total prompts: {len(prompts)}")
    logging.info(f"  FPS values per prompt: {len(fps_values)}")
    logging.info(f"  Total videos to generate: {len(prompts) * len(fps_values)}")

    os.makedirs(output_dir, exist_ok=True)

    all_results = []
    total_videos = len(prompts) * len(fps_values)
    video_count = 0

    for prompt_idx, (line_id, prompt_text, category) in enumerate(prompts, 1):
        logging.info(f"\n{'='*80}")
        logging.info(f"PROCESSING PROMPT {prompt_idx}/{len(prompts)}: {line_id:05d}")
        if category:
            logging.info(f"  Category: {category}")
        logging.info(f"  Prompt: {prompt_text}")
        logging.info(f"{'='*80}")

        # Per-prompt alignment calibration (--align mode)
        alignment_ratios = None
        if use_alignment:
            alignment_ratios = calibrate_alignment_ratios(
                pipeline=pipeline,
                config=config,
                prompt=prompt_text,
                n_prompt=generation_kwargs.get('n_prompt', ''),
                fps_values=fps_values,
                seed=generation_kwargs.get('seed', 42),
                steps=generation_kwargs.get('steps', 15),
                scale=generation_kwargs.get('scale', 7.0),
                frames=generation_kwargs.get('frames', 33),
                size=generation_kwargs.get('size', (832, 480)),
                shift=generation_kwargs.get('shift', 3.0),
            )
            if not alignment_ratios:
                logging.warning("  Calibration returned empty ratios — alignment disabled for this prompt")

            # Remove base LoRA weights so fps_only inference operates cleanly
            if alignment_ratios:
                logging.info("  Removing base LoRA parameters for fps_only inference...")
                base_lora_removed = 0
                for name, param in pipeline.transformer.named_parameters():
                    if '.lora_' in name and 'fps' not in name:
                        param.data.zero_()
                        base_lora_removed += 1
                logging.info(f"  Removed {base_lora_removed} base LoRA parameters")

        prompt_results = []

        for fps_idx, fps in enumerate(fps_values, 1):
            video_count += 1
            logging.info(f"\n--- Video {video_count}/{total_videos}: FPS = {fps} ---")

            try:
                video_tensor = generate_video_with_fps(
                    pipeline=pipeline,
                    config=config,
                    prompt=prompt_text,
                    n_prompt=generation_kwargs.get('n_prompt', ''),
                    fps=fps,
                    force_gate_one=force_gate_one,
                    alignment_ratios=alignment_ratios,
                    seed=generation_kwargs.get('seed', 42),
                    steps=generation_kwargs.get('steps', 15),
                    scale=generation_kwargs.get('scale', 7.0),
                    frames=generation_kwargs.get('frames', 33),
                    size=generation_kwargs.get('size', (832, 480)),
                    shift=generation_kwargs.get('shift', 3.0),
                )

                # Build filename
                fps_for_name = fps if include_fps_in_name else None
                filename = build_output_filename(line_id, prompt_text, fps_for_name)

                # Category subdirectory for --prompt_dir mode
                if category is not None:
                    save_dir = os.path.join(output_dir, category)
                else:
                    save_dir = output_dir

                os.makedirs(save_dir, exist_ok=True)
                filepath = os.path.join(save_dir, filename)

                logging.info(f"Saving: {filepath}")
                cache_video(tensor=video_tensor[None], save_file=filepath,
                            fps=16, normalize=True, value_range=(-1, 1))

                stats = {
                    'line_id': line_id,
                    'category': category,
                    'prompt': prompt_text,
                    'fps': fps,
                    'file': filepath,
                    'mean': video_tensor.mean().item(),
                    'std': video_tensor.std().item(),
                    'min': video_tensor.min().item(),
                    'max': video_tensor.max().item(),
                }
                prompt_results.append(stats)

                logging.info(f"Video {video_count}/{total_videos} complete: {os.path.basename(filepath)}")

            except Exception as e:
                logging.error(f"Video {video_count}/{total_videos} failed: {e}")
                import traceback
                traceback.print_exc()
                prompt_results.append({
                    'line_id': line_id,
                    'category': category,
                    'prompt': prompt_text,
                    'fps': fps,
                    'error': str(e),
                })

        all_results.append({
            'line_id': line_id,
            'category': category,
            'prompt': prompt_text,
            'results': prompt_results,
        })

        successful = len([r for r in prompt_results if 'error' not in r])
        logging.info(f"Prompt {line_id:05d} complete — {successful} videos generated")

    return all_results


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(all_results, output_dir):
    """Print summary statistics for all generated videos."""
    total_prompts = len(all_results)
    total_videos = sum(len(r['results']) for r in all_results)
    successful_videos = sum(len([v for v in r['results'] if 'error' not in v])
                            for r in all_results)
    failed_videos = total_videos - successful_videos

    logging.info(f"\n{'='*80}")
    logging.info(f"BATCH INFERENCE COMPLETE")
    logging.info(f"{'='*80}")
    logging.info(f"Total prompts processed: {total_prompts}")
    logging.info(f"Total videos generated: {successful_videos}/{total_videos}")
    if failed_videos > 0:
        logging.info(f"Failed videos: {failed_videos}")
    logging.info(f"Output directory: {output_dir}")

    categories = set(r['category'] for r in all_results if r['category'] is not None)
    if categories:
        logging.info(f"Results by category:")
        for category in sorted(categories):
            category_results = [r for r in all_results if r['category'] == category]
            category_videos = sum(len([v for v in r['results'] if 'error' not in v])
                                  for r in category_results)
            logging.info(f"  {category}: {len(category_results)} prompts, {category_videos} videos")

    logging.info(f"{'='*80}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Unified FPS conditioning inference for Less Is More (LIM)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single inline prompt, graft mode (paper method)
  PYTHONPATH=. python inference/inference.py \\
    --config configs/wan_SC_TARGET_14B_FPS_SHAPE_BLUR.toml \\
    --checkpoint checkpoints/20251014_06-32-03/epoch10 \\
    --prompt "A cheetah in full sprint across golden savannah" \\
    --output_dir output/single \\
    --condition_values 0.5 -1.0 0.0 \\
    --graft

  # Batch from file
  PYTHONPATH=. python inference/inference.py \\
    --config configs/wan_SC_TARGET_14B_FPS_SHAPE_BLUR.toml \\
    --checkpoint checkpoints/20251014_06-32-03/epoch10 \\
    --prompt_file prompts.txt \\
    --output_dir output/batch \\
    --condition_values 0.5 -1.0 0.0

  # Directory of category files (saves in subdirs per category)
  PYTHONPATH=. python inference/inference.py \\
    --config configs/wan_SC_TARGET_14B_FPS_SHAPE_BLUR.toml \\
    --checkpoint checkpoints/20251014_06-32-03/epoch10 \\
    --prompt_dir prompts/vbench/ \\
    --output_dir output/vbench \\
    --condition_values 0.5 -1.0 0.0

  # Prompt folder (one file per prompt)
  PYTHONPATH=. python inference/inference.py \\
    --config configs/wan_SC_TARGET_14B_FPS_SHAPE_BLUR.toml \\
    --checkpoint checkpoints/20251014_06-32-03/epoch10 \\
    --prompt_folder prompts/individual/ \\
    --output_dir output/individual \\
    --condition_values 0.5

  # Clean backbone (no LoRA)
  PYTHONPATH=. python inference/inference.py \\
    --config configs/wan_SC_TARGET_14B_FPS_SHAPE_BLUR.toml \\
    --prompt_file prompts.txt \\
    --output_dir output/clean \\
    --condition_values 0.5 \\
    --clean

  # Checkpoint sweep
  PYTHONPATH=. python inference/inference.py \\
    --config configs/wan_SC_TARGET_14B_FPS_SHAPE_BLUR.toml \\
    --checkpoint_parent checkpoints/20251014_06-32-03 \\
    --checkpoint_prefix epoch \\
    --checkpoint_start 5 \\
    --checkpoint_end 20 \\
    --checkpoint_interval 5 \\
    --prompt_file prompts.txt \\
    --output_dir output/sweep \\
    --condition_values 0.5 -1.0
        """
    )

    # Config (required)
    parser.add_argument('--config', required=True,
                        help='Path to TOML configuration file')

    # Checkpoint (single or sweep, mutually exclusive)
    checkpoint_group = parser.add_mutually_exclusive_group(required=False)
    checkpoint_group.add_argument('--checkpoint',
                                  help='Path to single checkpoint folder')
    checkpoint_group.add_argument('--checkpoint_parent',
                                  help='Parent folder containing checkpoint subfolders')

    parser.add_argument('--checkpoint_prefix', default='epoch',
                        help='Prefix for checkpoint subfolders (default: "epoch")')
    parser.add_argument('--checkpoint_start', type=int,
                        help='Starting checkpoint number (e.g. 5 for epoch5)')
    parser.add_argument('--checkpoint_end', type=int,
                        help='Ending checkpoint number (e.g. 20 for epoch20)')
    parser.add_argument('--checkpoint_interval', type=int, default=1,
                        help='Interval between checkpoints (default: 1)')

    parser.add_argument('--output_dir', required=True,
                        help='Output directory')

    # Prompt source (exactly one required)
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument('--prompt',
                              help='Single inline prompt string')
    prompt_group.add_argument('--prompt_file',
                              help='Path to .txt file with one prompt per line')
    prompt_group.add_argument('--prompt_dir',
                              help='Directory of .txt category files (one prompt per line each)')
    prompt_group.add_argument('--prompt_folder',
                              help='Directory of .txt files where each file = one prompt')

    # FPS and generation parameters
    parser.add_argument('--condition_values', nargs='+', type=float, default=[0.5],
                        help='Conditioning scalar values (range [-1, 1], anchor at 0.0). '
                             'Single-condition: space-separated scalars. '
                             'Multi-condition: groups of N values per experiment.')
    parser.add_argument('--negative_prompt', default='',
                        help='Negative prompt (optional)')
    parser.add_argument('--steps', type=int, default=15,
                        help='Number of denoising steps (default: 15)')
    parser.add_argument('--frames', type=int, default=33,
                        help='Number of video frames (default: 33)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--port', default='29501',
                        help='DeepSpeed distributed port (default: 29501)')
    parser.add_argument('--width', type=int, default=None,
                        help='Video width in pixels (default: None, uses TOML or 832)')
    parser.add_argument('--height', type=int, default=None,
                        help='Video height in pixels (default: None, uses TOML or 480)')
    parser.add_argument('--scale', type=float, default=None,
                        # 7.0 is a good general default; 6.0 works well for saturated conditioning values (e.g. ±1.5 range)
                        help='CFG guidance scale (default: None, falls back to 7.0)')

    # Inference mode flags (mutually exclusive)
    parser.add_argument('--graft', action='store_true',
                        help='GRAFT MODE: LoRA all blocks, FPS adapter in deepest-third only '
                             '(paper method)')
    parser.add_argument('--clean', action='store_true',
                        help='CLEAN MODE: pure backbone, no LoRA or FPS adapters loaded')
    parser.add_argument('--fps_only', action='store_true',
                        help='FPS-ONLY MODE: load only FPS-related parameters')
    parser.add_argument('--base_only', action='store_true',
                        help='BASE-ONLY MODE: load only base LoRA parameters')

    parser.add_argument('--align', action='store_true',
                        help='Alignment mode (requires --fps_only): calibrate magnitude ratios '
                             'per block before inference')
    parser.add_argument('--force_gate_one', action='store_true',
                        help='DIAGNOSTIC: force all FPS adapter gates to 1.0')

    args = parser.parse_args()

    # Validate mutually exclusive inference modes
    exclusive_modes = [args.fps_only, args.base_only, args.graft, args.clean]
    if sum(exclusive_modes) > 1:
        parser.error("--fps_only, --base_only, --graft, and --clean are mutually exclusive.")

    # Validate alignment requirements
    if args.align and not args.fps_only:
        parser.error("--align requires --fps_only mode.")

    # Validate checkpoint requirement
    if not args.clean and not args.checkpoint and not args.checkpoint_parent:
        parser.error("Either --checkpoint or --checkpoint_parent is required unless using --clean")

    # Validate checkpoint sweep arguments
    if args.checkpoint_parent:
        if args.checkpoint_start is None or args.checkpoint_end is None:
            parser.error("--checkpoint_parent requires both --checkpoint_start and --checkpoint_end")
        if args.checkpoint_start > args.checkpoint_end:
            parser.error("--checkpoint_start must be <= --checkpoint_end")
        if args.checkpoint_interval <= 0:
            parser.error("--checkpoint_interval must be positive")

    # Resolve resolution and scale defaults
    width = args.width if args.width is not None else 832
    height = args.height if args.height is not None else 480
    scale = args.scale if args.scale is not None else 7.0

    try:
        setup_environment(args.port)

        logging.info("Loading pipeline from TOML configuration...")
        pipeline, config = load_pipeline_from_toml(args.config)

        # Detect number of FPS conditions
        fps_tau_transform = config.get('model', {}).get('fps_tau_transform', 'log1p')
        if isinstance(fps_tau_transform, list):
            num_conditions = len(fps_tau_transform)
        else:
            num_conditions = 1

        logging.info(f"Detected {num_conditions}-condition model from config")

        condition_values_parsed = parse_condition_values(args.condition_values, num_conditions)
        logging.info(f"Parsed condition values: {condition_values_parsed}")

        # Determine checkpoint list
        if args.clean:
            checkpoints_to_process = [(None, None)]
        elif args.checkpoint:
            checkpoints_to_process = [(None, args.checkpoint)]
        elif args.checkpoint_parent:
            logging.info(f"Discovering checkpoints in: {args.checkpoint_parent}")
            logging.info(f"  Prefix: {args.checkpoint_prefix}, "
                         f"Range: [{args.checkpoint_start}, {args.checkpoint_end}], "
                         f"Interval: {args.checkpoint_interval}")
            checkpoints_to_process = discover_checkpoints(
                args.checkpoint_parent,
                args.checkpoint_prefix,
                args.checkpoint_start,
                args.checkpoint_end,
                args.checkpoint_interval,
            )
            logging.info(f"Found {len(checkpoints_to_process)} checkpoints to process")

        # Load prompts once (shared across all checkpoints)
        if args.prompt:
            logging.info("MODE: single inline prompt")
            prompts = load_prompts_from_inline(args.prompt)
        elif args.prompt_file:
            logging.info("MODE: prompt file (one prompt per line)")
            prompts = load_prompts_from_file(args.prompt_file)
        elif args.prompt_dir:
            logging.info("MODE: prompt directory (category files)")
            prompts = load_prompts_from_directory(args.prompt_dir)
        else:  # args.prompt_folder
            logging.info("MODE: prompt folder (one file per prompt)")
            prompts = load_prompts_from_folder(args.prompt_folder)

        # Process each checkpoint
        all_checkpoint_results = []
        for checkpoint_idx, (checkpoint_name, checkpoint_path) in enumerate(checkpoints_to_process, 1):
            if len(checkpoints_to_process) > 1:
                logging.info(f"\n{'#'*80}")
                logging.info(f"# CHECKPOINT {checkpoint_idx}/{len(checkpoints_to_process)}: {checkpoint_name}")
                logging.info(f"{'#'*80}\n")

                # Reload pipeline fresh for each checkpoint in sweep to avoid weight contamination
                if checkpoint_idx > 1 and not args.clean:
                    logging.info("Reloading pipeline fresh for new checkpoint...")
                    pipeline, config = load_pipeline_from_toml(args.config)

            # Determine output directory for this checkpoint
            if checkpoint_name is not None:
                checkpoint_output_dir = os.path.join(args.output_dir, checkpoint_name)
            else:
                checkpoint_output_dir = args.output_dir

            # Determine whether FPS value appears in filenames
            include_fps_in_name = True

            if args.clean:
                if checkpoint_idx == 1:
                    logging.info("CLEAN MODE: Using original backbone without any LoRA or FPS adapters")
                include_fps_in_name = False

            elif args.base_only:
                if checkpoint_idx == 1:
                    logging.info("BASE-ONLY MODE: Loading only base LoRA parameters")
                include_fps_in_name = False

                logging.info(f"Applying checkpoint: {checkpoint_path}")
                base_rank = config.get('adapter', {}).get('rank', 32)
                pipeline = apply_checkpoint(pipeline, checkpoint_path, rank=base_rank,
                                            fps_only=False, base_only=True,
                                            graft_mode=False, config=config)

            elif args.graft:
                if checkpoint_idx == 1:
                    validate_fps_config(config)
                    validate_base_lora_config(config)
                    fps_block_indices = get_fps_block_indices(config)
                    logging.info(f"GRAFT MODE: FPS + base LoRA in blocks {fps_block_indices}")

                logging.info(f"Applying checkpoint: {checkpoint_path}")
                base_rank = config.get('adapter', {}).get('rank', 32)
                pipeline = apply_checkpoint(pipeline, checkpoint_path, rank=base_rank,
                                            fps_only=False, base_only=False,
                                            graft_mode=True, config=config)
                include_fps_in_name = True

            else:
                # Normal, fps_only, or fps_only+align
                if args.fps_only:
                    if checkpoint_idx == 1:
                        validate_fps_config(config)
                    logging.info("FPS-ONLY MODE: Loading only FPS-related parameters")

                    if args.align:
                        # Load FULL checkpoint first (base LoRA + FPS) for calibration
                        logging.info("  Loading FULL checkpoint for alignment calibration...")
                        base_rank = config.get('adapter', {}).get('rank', 32)
                        pipeline = apply_checkpoint(pipeline, checkpoint_path, rank=base_rank,
                                                    fps_only=False, base_only=False,
                                                    graft_mode=False, config=config)
                    else:
                        logging.info(f"Applying checkpoint: {checkpoint_path}")
                        base_rank = config.get('adapter', {}).get('rank', 32)
                        pipeline = apply_checkpoint(pipeline, checkpoint_path, rank=base_rank,
                                                    fps_only=True, base_only=False,
                                                    graft_mode=False, config=config)
                else:
                    logging.info(f"Applying checkpoint: {checkpoint_path}")
                    base_rank = config.get('adapter', {}).get('rank', 32)
                    pipeline = apply_checkpoint(pipeline, checkpoint_path, rank=base_rank,
                                                fps_only=False, base_only=False,
                                                graft_mode=False, config=config)
                include_fps_in_name = True

            all_results = run_batch_inference(
                pipeline=pipeline,
                config=config,
                prompts=prompts,
                fps_values=condition_values_parsed,
                output_dir=checkpoint_output_dir,
                include_fps_in_name=include_fps_in_name,
                use_alignment=(args.align and args.fps_only),
                force_gate_one=args.force_gate_one,
                n_prompt=args.negative_prompt,
                seed=args.seed,
                steps=args.steps,
                frames=args.frames,
                size=(width, height),
                scale=scale,
                shift=3.0,
            )

            all_checkpoint_results.append({
                'checkpoint_name': checkpoint_name,
                'checkpoint_path': checkpoint_path,
                'output_dir': checkpoint_output_dir,
                'results': all_results,
            })

            if len(checkpoints_to_process) > 1:
                logging.info(f"Checkpoint {checkpoint_name} complete.")
                print_summary(all_results, checkpoint_output_dir)

        # Final summary
        if len(checkpoints_to_process) == 1:
            print_summary(all_checkpoint_results[0]['results'],
                          all_checkpoint_results[0]['output_dir'])
        else:
            logging.info(f"\n{'='*80}")
            logging.info(f"CHECKPOINT SWEEP COMPLETE")
            logging.info(f"{'='*80}")
            logging.info(f"Total checkpoints processed: {len(all_checkpoint_results)}")
            for ckpt_result in all_checkpoint_results:
                total_videos = sum(len(r['results']) for r in ckpt_result['results'])
                successful = sum(len([v for v in r['results'] if 'error' not in v])
                                 for r in ckpt_result['results'])
                logging.info(f"  {ckpt_result['checkpoint_name']}: {successful}/{total_videos} videos")
            logging.info(f"Output directory: {args.output_dir}")
            logging.info(f"{'='*80}\n")

    except Exception as e:
        logging.error(f"Script failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
