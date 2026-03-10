#!/usr/bin/env python3
"""
Batch FPS conditioning inference script with line-based prompt processing.
Supports two prompt reading modes:
1. Single txt file: Each line is a separate prompt
2. Directory with multiple txt files: Each file contains prompts (one per line)

Output naming:
- With FPS conditioning: {line_number}_{fps_condition}.mp4 (e.g., 00001_12fps.mp4)
- Without FPS conditioning (clean/base_only): {line_number}.mp4 (e.g., 00001.mp4)
"""

import torch
import os
import sys
import logging
from datetime import datetime
import deepspeed
from tqdm import tqdm
import math
import argparse
import toml
import json
from pathlib import Path


from models.wan.wan import WanPipeline
import peft
from inference_utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from inference_utils.utils import cache_video
from utils.common import DTYPE_MAP

# Import all the helper functions from the original script
from inference.test_fps_multiple_experiments_align import (
    setup_environment,
    load_pipeline_from_toml,
    detect_checkpoint_type,
    load_adapter_weights_selective,
    verify_fps_config_applied,
    apply_checkpoint,
    generate_video_with_fps,
    validate_fps_config,
    validate_base_lora_config,
    get_fps_block_indices
)
# Import parse_fps_values from the old file (not yet migrated to new file)
from inference.test_fps_multiple_experiments_align_old import parse_fps_values

def load_prompts_from_file(file_path):
    """
    Load prompts from a single text file where each line is a prompt.

    Args:
        file_path: Path to txt file with prompts (one per line)

    Returns:
        List of (line_number, prompt_text) tuples
        - line_number: 1-based line number formatted as 5-digit string (e.g., "00001")
        - prompt_text: The prompt text
    """
    file_path = Path(file_path)
    if not file_path.exists() or not file_path.is_file():
        raise ValueError(f"Prompt file does not exist or is not a file: {file_path}")

    prompts = []
    logging.info(f"\n📝 Loading prompts from file: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, start=1):
            prompt_text = line.strip()
            if prompt_text:  # Skip empty lines
                line_id = f"{line_num:05d}"  # Format as 00001, 00002, etc.
                prompts.append((line_id, prompt_text, None))  # (line_id, prompt_text, category)
                logging.info(f"  ✓ Line {line_id}: {prompt_text[:60]}...")

    if not prompts:
        raise ValueError(f"No valid prompts loaded from file: {file_path}")

    logging.info(f"📝 Total: {len(prompts)} prompts loaded\n")
    return prompts


def load_prompts_from_directory(dir_path):
    """
    Load prompts from multiple text files in a directory.
    Each txt file contains prompts (one per line).

    Args:
        dir_path: Path to directory containing .txt files

    Returns:
        List of (line_number, prompt_text, category) tuples
        - line_number: 1-based line number formatted as 5-digit string (e.g., "00001")
        - prompt_text: The prompt text
        - category: The txt file prefix (e.g., "animal" from "animal.txt")
    """
    dir_path = Path(dir_path)
    if not dir_path.exists() or not dir_path.is_dir():
        raise ValueError(f"Prompt directory does not exist or is not a directory: {dir_path}")

    txt_files = sorted(dir_path.glob('*.txt'))
    if not txt_files:
        raise ValueError(f"No .txt files found in directory: {dir_path}")

    prompts = []
    logging.info(f"\n📝 Loading prompts from directory: {dir_path}")
    logging.info(f"  Found {len(txt_files)} category files")

    for txt_file in txt_files:
        category = txt_file.stem  # e.g., "animal" from "animal.txt"
        logging.info(f"\n  📁 Category: {category}")

        with open(txt_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, start=1):
                prompt_text = line.strip()
                if prompt_text:  # Skip empty lines
                    line_id = f"{line_num:05d}"  # Format as 00001, 00002, etc.
                    prompts.append((line_id, prompt_text, category))
                    if line_num <= 3:  # Show first 3 prompts per category
                        logging.info(f"    ✓ Line {line_id}: {prompt_text[:60]}...")

        logging.info(f"    Loaded {sum(1 for _, _, c in prompts if c == category)} prompts from {category}")

    if not prompts:
        raise ValueError(f"No valid prompts loaded from directory: {dir_path}")

    logging.info(f"\n📝 Total: {len(prompts)} prompts loaded from {len(txt_files)} categories\n")
    return prompts


def format_fps_for_filename(fps):
    """
    Format FPS value(s) for use in filename.

    Args:
        fps: Scalar float or list of floats

    Returns:
        String suitable for filename (e.g., "12fps", "0_50_0_02fps" for multi-condition)
    """
    if isinstance(fps, list):
        # Multi-condition: join with underscore, e.g., [0.5, 0.02] -> "0_50_0_02fps"
        fps_str = '_'.join(f"{v:.2f}".replace('.', '_') for v in fps)
        return f"{fps_str}fps"
    else:
        # Single-condition: e.g., 12.0 -> "12fps"
        if fps == int(fps):
            return f"{int(fps)}fps"
        else:
            return f"{fps:.2f}".replace('.', '_') + "fps"


def save_video_with_line_number(tensor, line_id, category, fps, output_dir, include_fps_in_name, size):
    """
    Saves video with line-number-based filename.

    Args:
        tensor: Video tensor to save
        line_id: Line number as string (e.g., "00001")
        category: Category name (None for single file mode)
        fps: FPS value (scalar or list) used for generation
        output_dir: Base output directory
        include_fps_in_name: Whether to include FPS condition in filename
        size: Video size tuple (width, height)

    Returns:
        Path to saved video file
    """
    if tensor is None:
        logging.warning("Video tensor is None, skipping save.")
        return None

    # Create category subfolder if in directory mode
    if category is not None:
        save_dir = os.path.join(output_dir, category)
    else:
        save_dir = output_dir

    os.makedirs(save_dir, exist_ok=True)

    # Build filename
    if include_fps_in_name:
        fps_str = format_fps_for_filename(fps)
        filename = f"{line_id}_{fps_str}.mp4"
    else:
        filename = f"{line_id}.mp4"

    filepath = os.path.join(save_dir, filename)

    logging.info(f"💾 Saving: {filepath}")
    cache_video(tensor=tensor[None], save_file=filepath, fps=16, normalize=True, value_range=(-1, 1))

    return filepath


def run_batch_inference(pipeline, config, prompts, fps_values, output_dir, include_fps_in_name,
                       force_gate_one=False, **generation_kwargs):
    """
    Runs batch inference for all prompts and FPS values.

    Args:
        pipeline: WanPipeline instance
        config: TOML configuration dict
        prompts: List of (line_id, prompt_text, category) tuples
        fps_values: List of FPS values to test (parsed based on num_conditions)
        output_dir: Base output directory
        include_fps_in_name: Whether to include FPS in filename (False for clean/base_only modes)
        force_gate_one: Force FPS adapter gates to 1.0
        **generation_kwargs: Additional generation parameters (seed, steps, frames, size, etc.)

    Returns:
        List of results for all prompts and FPS values
    """
    logging.info("🚀 Starting batch inference...")
    logging.info(f"  Total prompts: {len(prompts)}")
    logging.info(f"  FPS values per prompt: {len(fps_values)}")
    logging.info(f"  Total videos to generate: {len(prompts) * len(fps_values)}")
    logging.info(f"  Include FPS in filename: {include_fps_in_name}")

    os.makedirs(output_dir, exist_ok=True)

    all_results = []
    total_videos = len(prompts) * len(fps_values)
    video_count = 0

    for prompt_idx, (line_id, prompt_text, category) in enumerate(prompts, 1):
        logging.info(f"\n{'='*80}")
        logging.info(f"PROCESSING PROMPT {prompt_idx}/{len(prompts)}: Line {line_id}")
        if category:
            logging.info(f"  Category: {category}")
        logging.info(f"  Prompt: {prompt_text}")
        logging.info(f"{'='*80}")

        prompt_results = []

        for fps_idx, fps in enumerate(fps_values, 1):
            video_count += 1
            logging.info(f"\n--- Video {video_count}/{total_videos}: FPS = {fps} ---")

            try:
                # Generate video
                video_tensor = generate_video_with_fps(
                    pipeline=pipeline,
                    config=config,
                    prompt=prompt_text,
                    n_prompt=generation_kwargs.get('n_prompt', ''),
                    fps=fps,
                    force_gate_one=force_gate_one,
                    seed=generation_kwargs.get('seed', 42),
                    steps=generation_kwargs.get('steps', 15),
                    scale=generation_kwargs.get('scale', 6.0),
                    frames=generation_kwargs.get('frames', 33),
                    size=generation_kwargs.get('size', (832, 480)),
                    shift=generation_kwargs.get('shift', 3.0)
                )

                # Save result with line-number-based naming
                output_file = save_video_with_line_number(
                    tensor=video_tensor,
                    line_id=line_id,
                    category=category,
                    fps=fps,
                    output_dir=output_dir,
                    include_fps_in_name=include_fps_in_name,
                    size=generation_kwargs.get('size', (832, 480))
                )

                # Collect stats
                stats = {
                    'line_id': line_id,
                    'category': category,
                    'prompt': prompt_text,
                    'fps': fps,
                    'file': output_file,
                    'mean': video_tensor.mean().item(),
                    'std': video_tensor.std().item(),
                    'min': video_tensor.min().item(),
                    'max': video_tensor.max().item()
                }
                prompt_results.append(stats)

                logging.info(f"✅ Video {video_count}/{total_videos} complete: {os.path.basename(output_file)}")

            except Exception as e:
                logging.error(f"❌ Video {video_count}/{total_videos} failed: {e}")
                import traceback
                traceback.print_exc()
                prompt_results.append({
                    'line_id': line_id,
                    'category': category,
                    'prompt': prompt_text,
                    'fps': fps,
                    'error': str(e)
                })

        all_results.append({
            'line_id': line_id,
            'category': category,
            'prompt': prompt_text,
            'results': prompt_results
        })

        logging.info(f"\n✅ Prompt {line_id} complete! Generated {len([r for r in prompt_results if 'error' not in r])} videos")

    return all_results


def discover_checkpoints(parent_folder, prefix, start, end, interval):
    """
    Discover checkpoint subfolders based on prefix and range.

    Args:
        parent_folder: Parent directory containing checkpoint subfolders
        prefix: Prefix for checkpoint folders (e.g., "epoch")
        start: Starting checkpoint number
        end: Ending checkpoint number
        interval: Interval between checkpoints

    Returns:
        List of (checkpoint_name, checkpoint_path) tuples
    """
    from pathlib import Path

    parent_path = Path(parent_folder)
    if not parent_path.exists() or not parent_path.is_dir():
        raise ValueError(f"Checkpoint parent folder does not exist or is not a directory: {parent_folder}")

    checkpoints = []
    for checkpoint_num in range(start, end + 1, interval):
        checkpoint_name = f"{prefix}{checkpoint_num}"
        checkpoint_path = parent_path / checkpoint_name

        if checkpoint_path.exists() and checkpoint_path.is_dir():
            # Check if there's at least one .safetensors file
            safetensors_files = list(checkpoint_path.glob('*.safetensors'))
            if safetensors_files:
                checkpoints.append((checkpoint_name, str(checkpoint_path)))
                logging.info(f"  ✓ Found checkpoint: {checkpoint_name}")
            else:
                logging.warning(f"  ⚠ Skipping {checkpoint_name}: no .safetensors files found")
        else:
            logging.warning(f"  ✗ Skipping {checkpoint_name}: folder does not exist")

    if not checkpoints:
        raise ValueError(
            f"No valid checkpoints found in {parent_folder} with prefix '{prefix}' "
            f"and range [{start}, {end}] interval {interval}"
        )

    return checkpoints


def print_summary(all_results, output_dir):
    """Print summary statistics for all generated videos."""
    total_prompts = len(all_results)
    total_videos = sum(len(r['results']) for r in all_results)
    successful_videos = sum(len([v for v in r['results'] if 'error' not in v]) for r in all_results)
    failed_videos = total_videos - successful_videos

    logging.info(f"\n{'='*80}")
    logging.info(f"🎉 BATCH INFERENCE COMPLETE!")
    logging.info(f"{'='*80}")
    logging.info(f"Total prompts processed: {total_prompts}")
    logging.info(f"Total videos generated: {successful_videos}/{total_videos}")
    if failed_videos > 0:
        logging.info(f"Failed videos: {failed_videos}")
    logging.info(f"Output directory: {output_dir}")

    # Group by category if applicable
    categories = set(r['category'] for r in all_results if r['category'] is not None)
    if categories:
        logging.info(f"\n📁 Results by category:")
        for category in sorted(categories):
            category_results = [r for r in all_results if r['category'] == category]
            category_videos = sum(len([v for v in r['results'] if 'error' not in v]) for r in category_results)
            logging.info(f"  {category}: {len(category_results)} prompts, {category_videos} videos")

    logging.info(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Batch FPS conditioning inference with line-based prompts',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Mode 1: Single txt file with prompts (one per line)
  python test_fps_batch_prompts.py \\
    --config bokeh_TOML/wan_SC_TARGET_14B_3SHAPES_BASE.toml \\
    --checkpoint checkpoints/20251026_07-09-05/epoch700 \\
    --prompt_file utils/VBench/prompts/all_category.txt \\
    --output_dir output/vbench_all \\
    --fps_values 12 24 60

  # Mode 2: Directory with multiple txt files (each with prompts)
  python test_fps_batch_prompts.py \\
    --config bokeh_TOML/wan_SC_TARGET_14B_3SHAPES_BASE.toml \\
    --checkpoint checkpoints/20251026_07-09-05/epoch700 \\
    --prompt_dir utils/VBench/prompts/prompts_per_category \\
    --output_dir output/vbench_by_category \\
    --fps_values 12 24 60

  # Checkpoint sweep: Test multiple checkpoints (epoch100, epoch200, ..., epoch1000)
  python test_fps_batch_prompts.py \\
    --config checkpoints/20251008_20-21-45/wan_SC_TARGET_14B_3SHAPES.toml \\
    --checkpoint_parent checkpoints/20251008_20-21-45 \\
    --checkpoint_prefix epoch \\
    --checkpoint_start 100 \\
    --checkpoint_end 1000 \\
    --checkpoint_interval 100 \\
    --prompt_file utils/VBench/prompts/all_category.txt \\
    --output_dir output/checkpoint_sweep \\
    --fps_values 12 24 60
    # Output: output/checkpoint_sweep/epoch100/00001_12fps.mp4
    #         output/checkpoint_sweep/epoch200/00001_12fps.mp4
    #         ...

  # Clean mode (no LoRA, no FPS in filename)
  python test_fps_batch_prompts.py \\
    --config bokeh_TOML/wan_SC_TARGET_14B_3SHAPES_BASE.toml \\
    --clean \\
    --prompt_file utils/VBench/prompts/all_category.txt \\
    --output_dir output/vbench_clean \\
    --fps_values 24
        """
    )

    # Required arguments
    parser.add_argument('--config', required=True, help='Path to TOML configuration file')

    # Checkpoint arguments (mutually exclusive groups)
    checkpoint_group = parser.add_mutually_exclusive_group(required=False)
    checkpoint_group.add_argument('--checkpoint', help='Path to single checkpoint folder (not needed if using --clean)')
    checkpoint_group.add_argument('--checkpoint_parent', help='Parent folder containing multiple checkpoint subfolders')

    # Checkpoint sweep arguments (only used with --checkpoint_parent)
    parser.add_argument('--checkpoint_prefix', default='epoch', help='Prefix for checkpoint subfolders (default: "epoch")')
    parser.add_argument('--checkpoint_start', type=int, help='Starting checkpoint number (e.g., 100 for epoch100)')
    parser.add_argument('--checkpoint_end', type=int, help='Ending checkpoint number (e.g., 1000 for epoch1000)')
    parser.add_argument('--checkpoint_interval', type=int, default=100, help='Interval between checkpoints (default: 100)')

    parser.add_argument('--output_dir', required=True, help='Output directory')

    # Prompt source (exactly one required)
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument('--prompt_file', help='Mode 1: Path to single txt file with prompts (one per line)')
    prompt_group.add_argument('--prompt_dir', help='Mode 2: Path to directory with multiple txt files')

    # FPS and generation parameters
    parser.add_argument('--fps_values', nargs='+', type=float, default=[12, 24, 60],
                       help='FPS values to test. Single-condition: --fps_values 12 24 60. ' +
                            'Multi-condition (2): --fps_values 0.5 0.02  1.0 0.05  0.125 0.1')
    parser.add_argument('--negative_prompt', default='', help='Negative prompt (optional)')
    parser.add_argument('--steps', type=int, default=15, help='Denoising steps')
    parser.add_argument('--frames', type=int, default=33, help='Number of frames')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--port', default='29501', help='DeepSpeed port')
    parser.add_argument('--width', type=int, default=832, help='Video width')
    parser.add_argument('--height', type=int, default=480, help='Video height')
    parser.add_argument('--scale', type=float, default=6.0, help='CFG scale')

    # Loading modes
    parser.add_argument('--force_gate_one', action='store_true',
                       help='🔧 DIAGNOSTIC: Force all FPS adapter gates to 1.0 for maximum FPS impact')
    parser.add_argument('--fps_only', action='store_true',
                       help='Load only FPS-related parameters from checkpoint (ignore base LoRA)')
    parser.add_argument('--base_only', action='store_true',
                       help='Load only base LoRA parameters from checkpoint (ignore FPS parameters)')
    parser.add_argument('--graft', action='store_true',
                       help='🌿 GRAFT MODE: Load base LoRA only in blocks where FPS adapters exist (spatial matching)')
    parser.add_argument('--clean', action='store_true',
                       help='Use clean backbone without loading any LoRA (neither base LoRA nor FPS adapters)')

    args = parser.parse_args()

    # Validate selective loading arguments
    exclusive_modes = [args.fps_only, args.base_only, args.graft, args.clean]
    if sum(exclusive_modes) > 1:
        parser.error("--fps_only, --base_only, --graft, and --clean are mutually exclusive. Choose at most one.")

    # Validate checkpoint requirement
    if not args.clean and not args.checkpoint and not args.checkpoint_parent:
        parser.error("Either --checkpoint or --checkpoint_parent is required unless using --clean mode")

    # Validate checkpoint_parent arguments
    if args.checkpoint_parent:
        if not args.checkpoint_start or not args.checkpoint_end:
            parser.error("--checkpoint_parent requires both --checkpoint_start and --checkpoint_end")
        if args.checkpoint_start > args.checkpoint_end:
            parser.error("--checkpoint_start must be <= --checkpoint_end")
        if args.checkpoint_interval <= 0:
            parser.error("--checkpoint_interval must be positive")

    try:
        # Setup
        setup_environment(args.port)

        # Load pipeline using TOML configuration
        logging.info("Loading pipeline from TOML configuration...")
        pipeline, config = load_pipeline_from_toml(args.config)

        # Detect number of conditions from config
        fps_tau_transform = config.get('model', {}).get('fps_tau_transform', 'log1p')
        if isinstance(fps_tau_transform, list):
            num_conditions = len(fps_tau_transform)
        else:
            num_conditions = 1

        logging.info(f"📊 Detected {num_conditions}-condition model from config")

        # Parse fps_values based on num_conditions
        fps_values_parsed = parse_fps_values(args.fps_values, num_conditions)
        logging.info(f"📊 Parsed FPS values from: {args.fps_values}")
        if num_conditions == 1:
            logging.info(f"  Single-condition: {fps_values_parsed}")
        else:
            logging.info(f"  Multi-condition ({num_conditions} conditions per experiment):")
            for i, vals in enumerate(fps_values_parsed, 1):
                logging.info(f"    Experiment {i}: [{', '.join(str(v) for v in vals)}]")

        # Determine checkpoint list: single checkpoint or multiple checkpoints
        if args.clean:
            # Clean mode: no checkpoints
            checkpoints_to_process = [(None, None)]  # Single pass with no checkpoint
        elif args.checkpoint:
            # Single checkpoint mode
            checkpoints_to_process = [(None, args.checkpoint)]  # (name, path)
        elif args.checkpoint_parent:
            # Checkpoint sweep mode
            logging.info(f"\n📂 Discovering checkpoints in: {args.checkpoint_parent}")
            logging.info(f"  Prefix: {args.checkpoint_prefix}")
            logging.info(f"  Range: [{args.checkpoint_start}, {args.checkpoint_end}]")
            logging.info(f"  Interval: {args.checkpoint_interval}")
            checkpoints_to_process = discover_checkpoints(
                args.checkpoint_parent,
                args.checkpoint_prefix,
                args.checkpoint_start,
                args.checkpoint_end,
                args.checkpoint_interval
            )
            logging.info(f"\n✅ Found {len(checkpoints_to_process)} checkpoints to process\n")

        # Load prompts once (shared across all checkpoints)
        if args.prompt_file:
            logging.info("📝 MODE 1: Loading prompts from single file")
            prompts = load_prompts_from_file(args.prompt_file)
        else:  # args.prompt_dir
            logging.info("📝 MODE 2: Loading prompts from directory")
            prompts = load_prompts_from_directory(args.prompt_dir)

        # Process each checkpoint
        all_checkpoint_results = []
        for checkpoint_idx, (checkpoint_name, checkpoint_path) in enumerate(checkpoints_to_process, 1):
            if len(checkpoints_to_process) > 1:
                logging.info(f"\n{'#'*80}")
                logging.info(f"# PROCESSING CHECKPOINT {checkpoint_idx}/{len(checkpoints_to_process)}: {checkpoint_name}")
                logging.info(f"{'#'*80}\n")

                # IMPORTANT: Reload pipeline fresh for each checkpoint in sweep mode
                # This prevents parameter conflicts from previous checkpoint
                if checkpoint_idx > 1 and not args.clean:
                    logging.info("🔄 Reloading pipeline fresh for new checkpoint...")
                    pipeline, config = load_pipeline_from_toml(args.config)

            # Determine output directory for this checkpoint
            if checkpoint_name is not None:
                # Checkpoint sweep: add checkpoint subfolder
                checkpoint_output_dir = os.path.join(args.output_dir, checkpoint_name)
            else:
                # Single checkpoint or clean mode: use base output dir
                checkpoint_output_dir = args.output_dir

            # Validate selective loading requirements and apply checkpoint
            include_fps_in_name = True  # Default: include FPS in filename

            if args.clean:
                if checkpoint_idx == 1:  # Only log once for clean mode
                    logging.info("🧹 CLEAN MODE: Using original backbone without any LoRA")
                    logging.info("   → Skipping checkpoint loading entirely")
                    logging.info("   → No base LoRA will be applied")
                    logging.info("   → No FPS adapters will be applied")
                    logging.info("   → Output filenames will NOT include FPS condition")
                include_fps_in_name = False
            elif args.base_only:
                if checkpoint_idx == 1:  # Only log once
                    logging.info("🎯 BASE-ONLY MODE: Will load only base LoRA parameters")
                    logging.info("   → Output filenames will NOT include FPS condition")
                include_fps_in_name = False

                # Apply checkpoint
                logging.info(f"Applying checkpoint: {checkpoint_path}")
                base_rank = config.get('adapter', {}).get('rank', 32)
                pipeline = apply_checkpoint(pipeline, checkpoint_path, rank=base_rank,
                                           fps_only=args.fps_only, base_only=args.base_only,
                                           graft_mode=False, config=config)
            elif args.graft:
                if checkpoint_idx == 1:  # Only validate once
                    validate_fps_config(config)
                    validate_base_lora_config(config)
                    fps_block_indices = get_fps_block_indices(config)
                    logging.info(f"🌿 GRAFT MODE: Will load FPS parameters + base LoRA only in blocks {fps_block_indices}")
                    logging.info("   → Output filenames WILL include FPS condition")

                # Apply checkpoint
                logging.info(f"Applying checkpoint: {checkpoint_path}")
                base_rank = config.get('adapter', {}).get('rank', 32)
                pipeline = apply_checkpoint(pipeline, checkpoint_path, rank=base_rank,
                                           fps_only=False, base_only=False,
                                           graft_mode=True, config=config)
                include_fps_in_name = True
            else:
                # Normal mode or FPS-only mode
                if args.fps_only:
                    if checkpoint_idx == 1:  # Only validate once
                        validate_fps_config(config)
                    logging.info("🎯 FPS-ONLY MODE: Will load only FPS-related parameters")

                # Apply checkpoint
                logging.info(f"Applying checkpoint: {checkpoint_path}")
                base_rank = config.get('adapter', {}).get('rank', 32)
                pipeline = apply_checkpoint(pipeline, checkpoint_path, rank=base_rank,
                                           fps_only=args.fps_only, base_only=args.base_only,
                                           graft_mode=False, config=config)
                include_fps_in_name = True

            # Run batch inference for this checkpoint
            all_results = run_batch_inference(
                pipeline=pipeline,
                config=config,
                prompts=prompts,
                fps_values=fps_values_parsed,
                output_dir=checkpoint_output_dir,
                include_fps_in_name=include_fps_in_name,
                force_gate_one=args.force_gate_one,
                n_prompt=args.negative_prompt,
                seed=args.seed,
                steps=args.steps,
                frames=args.frames,
                size=(args.width, args.height),
                scale=args.scale,
                shift=3.0
            )

            # Store results for this checkpoint
            all_checkpoint_results.append({
                'checkpoint_name': checkpoint_name,
                'checkpoint_path': checkpoint_path,
                'output_dir': checkpoint_output_dir,
                'results': all_results
            })

            # Print summary for this checkpoint
            if len(checkpoints_to_process) > 1:
                logging.info(f"\n✅ Checkpoint {checkpoint_name} complete!")
                print_summary(all_results, checkpoint_output_dir)

        # Print final summary
        if len(checkpoints_to_process) == 1:
            # Single checkpoint: print regular summary
            print_summary(all_checkpoint_results[0]['results'], all_checkpoint_results[0]['output_dir'])
        else:
            # Multiple checkpoints: print sweep summary
            logging.info(f"\n{'='*80}")
            logging.info(f"🎉 CHECKPOINT SWEEP COMPLETE!")
            logging.info(f"{'='*80}")
            logging.info(f"Total checkpoints processed: {len(all_checkpoint_results)}")
            for ckpt_result in all_checkpoint_results:
                total_videos = sum(len(r['results']) for r in ckpt_result['results'])
                successful = sum(len([v for v in r['results'] if 'error' not in v]) for r in ckpt_result['results'])
                logging.info(f"  {ckpt_result['checkpoint_name']}: {successful}/{total_videos} videos")
            logging.info(f"Output directory: {args.output_dir}")
            logging.info(f"{'='*80}\n")

    except Exception as e:
        logging.error(f"❌ Script failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
