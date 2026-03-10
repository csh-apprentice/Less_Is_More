#!/usr/bin/env python3
"""
Multi-experiment FPS conditioning test script - TOML-aligned version.
Tests different FPS values to verify FPS conditioning works correctly.
Uses TOML configuration files for proper training/inference alignment.
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


from models.wan.wan import WanPipeline
import peft
from inference_utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from inference_utils.utils import cache_video
from utils.common import DTYPE_MAP

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
    
    # Load TOML exactly like train.py does
    with open(toml_path) as f:
        config = json.loads(json.dumps(toml.load(f)))
    
    # Convert dtype strings to torch.dtype objects (like train.py does)
    model_dtype_str = config['model']['dtype']
    config['model']['dtype'] = DTYPE_MAP[model_dtype_str]
    if transformer_dtype := config['model'].get('transformer_dtype', None):
        config['model']['transformer_dtype'] = DTYPE_MAP.get(transformer_dtype, transformer_dtype)
    
    logging.info("Initializing WanPipeline with TOML configuration...")
    logging.info(f"Model type: {config['model']['type']}")
    logging.info(f"Model checkpoint: {config['model']['ckpt_path']}")
    logging.info(f"Model dtype: {model_dtype_str} -> {config['model']['dtype']}")
    
    # Log ALL FPS configuration from TOML (must match wan.py exactly)
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
        'fps_warmup_steps': model_config.get('fps_warmup_steps')
    }
    logging.info("Complete FPS configuration from TOML (aligned with wan.py):")
    for key, value in fps_settings.items():
        if value is not None:
            logging.info(f"  {key}: {value}")
        else:
            # Show which settings are missing and will use defaults
            logging.info(f"  {key}: <not set, will use wan.py default>")
    
    # Initialize pipeline using TOML config (same as train.py)
    wan_t2v = WanPipeline(config)
    
    logging.info("Loading main transformer model...")
    wan_t2v.load_diffusion_model()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Moving model components to device: {device}")
    
    wan_t2v.transformer.to(device)
    wan_t2v.vae.model.to(device)
    wan_t2v.text_encoder.model.to(device)
    
    # SUBTASK 3: Set models to eval mode for sampling (critical for proper inference)
    logging.info("Setting models to eval mode for inference...")
    wan_t2v.transformer.eval()
    wan_t2v.vae.model.eval()
    wan_t2v.text_encoder.model.eval()
    
    logging.info("Pipeline loaded successfully with TOML configuration.")
    return wan_t2v, config

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

def load_adapter_weights_selective(pipeline, checkpoint_path, fps_only=False, base_only=False):
    """Load adapter weights with selective parameter filtering using temporary files."""
    if not fps_only and not base_only:
        # Normal mode - use original loading method
        logging.info(f"📦 NORMAL MODE: Loading all parameters")
        pipeline.load_adapter_weights(checkpoint_path)
        return
    
    import safetensors
    import tempfile
    from pathlib import Path
    
    # Use the same checkpoint detection logic as the original WAN pipeline
    checkpoint_dir = Path(checkpoint_path)
    if checkpoint_dir.is_dir():
        # Look for .safetensors files in the directory
        safetensors_files = list(checkpoint_dir.glob('*.safetensors'))
        if not safetensors_files:
            raise FileNotFoundError(f"No .safetensors files found in {checkpoint_dir}")
        checkpoint_file = safetensors_files[0]  # Use the first one found
        logging.info(f"Found checkpoint file: {checkpoint_file}")
    else:
        checkpoint_file = checkpoint_dir
    
    # Load the original checkpoint
    logging.info(f"Loading checkpoint: {checkpoint_file}")
    checkpoint = safetensors.torch.load_file(str(checkpoint_file))
    
    # Filter parameters based on mode
    if fps_only:
        # Only keep FPS-related parameters
        filtered_checkpoint = {k: v for k, v in checkpoint.items() 
                             if 'fps_conditioning' in k or 'fps_adapter' in k}
        mode_name = "FPS-ONLY"
        logging.info(f"🎯 {mode_name} MODE: Filtered {len(filtered_checkpoint)} FPS parameters out of {len(checkpoint)} total")
        
    elif base_only:
        # Only keep base LoRA parameters (exclude FPS parameters)
        filtered_checkpoint = {k: v for k, v in checkpoint.items() 
                             if 'fps_conditioning' not in k and 'fps_adapter' not in k}
        mode_name = "BASE-ONLY"
        logging.info(f"🎯 {mode_name} MODE: Filtered {len(filtered_checkpoint)} base LoRA parameters out of {len(checkpoint)} total")
    
    # Show detailed breakdown of what's being loaded/skipped
    fps_conditioning_params = [k for k in checkpoint.keys() if 'fps_conditioning' in k]
    fps_adapter_params = [k for k in checkpoint.keys() if 'fps_adapter' in k]
    base_lora_params = [k for k in checkpoint.keys() if 'fps_conditioning' not in k and 'fps_adapter' not in k]
    
    logging.info(f"📊 Parameter breakdown in original checkpoint:")
    logging.info(f"  FPS conditioning (MLP): {len(fps_conditioning_params)} parameters")
    logging.info(f"  FPS adapters (LoRA): {len(fps_adapter_params)} parameters") 
    logging.info(f"  Base LoRA: {len(base_lora_params)} parameters")
    logging.info(f"  Total: {len(checkpoint)} parameters")
    
    if fps_only:
        logging.info(f"🎯 FPS-ONLY filtering results:")
        logging.info(f"  ✅ Loading FPS conditioning: {len(fps_conditioning_params)} parameters")
        logging.info(f"  ✅ Loading FPS adapters: {len(fps_adapter_params)} parameters")
        logging.info(f"  ❌ Skipping base LoRA: {len(base_lora_params)} parameters")
        
        # Show examples of each type
        if fps_conditioning_params:
            logging.info(f"  📝 Example FPS conditioning params: {fps_conditioning_params[:2]}")
        if fps_adapter_params:
            logging.info(f"  📝 Example FPS adapter params: {fps_adapter_params[:2]}")
            
    elif base_only:
        logging.info(f"🎯 BASE-ONLY filtering results:")
        logging.info(f"  ✅ Loading base LoRA: {len(base_lora_params)} parameters") 
        logging.info(f"  ❌ Skipping FPS conditioning: {len(fps_conditioning_params)} parameters")
        logging.info(f"  ❌ Skipping FPS adapters: {len(fps_adapter_params)} parameters")
        
        # Show examples of base LoRA params
        if base_lora_params:
            logging.info(f"  📝 Example base LoRA params: {base_lora_params[:3]}")
    
    # Create temporary directory and filtered safetensors file
    temp_dir = tempfile.mkdtemp(prefix=f"selective_loading_{mode_name.lower()}_")
    temp_checkpoint_file = Path(temp_dir) / "filtered_adapter.safetensors"
    
    logging.info(f"Creating temporary filtered checkpoint: {temp_checkpoint_file}")
    safetensors.torch.save_file(filtered_checkpoint, str(temp_checkpoint_file))
    
    # Use the original load_adapter_weights method with the filtered temporary file
    try:
        pipeline.load_adapter_weights(temp_dir)
        logging.info(f"✅ Successfully loaded {len(filtered_checkpoint)} filtered parameters using temporary file")
    finally:
        # Clean up temporary files
        import shutil
        shutil.rmtree(temp_dir)
        logging.info(f"🧹 Cleaned up temporary directory: {temp_dir}")

def verify_fps_config_applied(pipeline, config):
    """Verify that FPS configuration from TOML was properly applied to the model."""
    model_config = config['model']
    logging.info("🔍 Verifying FPS configuration from TOML was applied correctly:")
    
    # Check FPS adapters exist in the model
    fps_adapter_count = sum(1 for n, _ in pipeline.transformer.named_parameters() if 'fps_adapter' in n)
    fps_conditioning_count = sum(1 for n, _ in pipeline.transformer.named_parameters() if 'fps_conditioning' in n)
    
    logging.info(f"  Found {fps_adapter_count} FPS adapter parameters in model")
    logging.info(f"  Found {fps_conditioning_count} FPS conditioning (MLP) parameters in model")
    
    # Check if the expected FPS settings were used
    expected_settings = {
        'fps_adapter_rank': model_config.get('fps_adapter_rank', 'default'),
        'fps_condition_blocks': model_config.get('fps_condition_blocks', 'default'),
        'fps_gate_mode': model_config.get('fps_gate_mode', 'default'),
        'fps_gate_fixed_value': model_config.get('fps_gate_fixed_value', 'default'),
        'fps_adapter_num_tokens': model_config.get('fps_adapter_num_tokens', 'default'),
        'fps_embed_dim': model_config.get('fps_embed_dim', 'default'),
        'fps_reference_fps': model_config.get('fps_reference_fps', 'default (240.0)'),
        'fps_scale': model_config.get('fps_scale', 'default (False)'),
        'fps_warmup_steps': model_config.get('fps_warmup_steps', 'default (100)')
    }
    
    logging.info("  Expected FPS configuration from TOML:")
    for key, value in expected_settings.items():
        logging.info(f"    {key}: {value}")
    
    # Check gate mode by looking at buffer vs parameter names
    has_gate_alpha = any('gate_alpha' in n for n, _ in pipeline.transformer.named_parameters())
    has_gate_fixed = any('gate_fixed' in n for n, _ in pipeline.transformer.named_buffers())

    # Learnable modes: sigmoid, identity, relu, silu, softplus (all use gate_alpha parameter)
    # Fixed mode: uses gate_fixed buffer
    if has_gate_alpha and not has_gate_fixed:
        detected_mode = "learnable (sigmoid/identity/relu/silu/softplus)"
    elif has_gate_fixed and not has_gate_alpha:
        detected_mode = "fixed"
    elif has_gate_alpha and has_gate_fixed:
        detected_mode = "mixed (both found)"
    else:
        detected_mode = "unknown"

    expected_mode = model_config.get('fps_gate_mode', 'sigmoid')  # Default changed from 'learned' to 'sigmoid'
    # Check if expected mode matches detected mode category
    learnable_modes = ['sigmoid', 'identity', 'relu', 'silu', 'softplus']
    expected_is_learnable = expected_mode in learnable_modes
    detected_is_learnable = has_gate_alpha and not has_gate_fixed
    mode_match = "✅" if (expected_is_learnable == detected_is_learnable) or (expected_mode == 'fixed' and detected_mode == 'fixed') else "⚠️"
    logging.info(f"  Gate mode check: Expected='{expected_mode}', Detected='{detected_mode}' {mode_match}")

    # Verify fps_reference_fps is being used correctly
    actual_reference_fps = getattr(pipeline, 'fps_reference_fps', 'not found')
    expected_reference_fps = model_config.get('fps_reference_fps', 240.0)
    reference_match = "✅" if actual_reference_fps == expected_reference_fps else "⚠️"
    logging.info(f"  Reference FPS check: Expected={expected_reference_fps}, Actual={actual_reference_fps} {reference_match}")

def apply_checkpoint(wan_t2v_pipeline, checkpoint_path, rank=32, dtype=torch.bfloat16, fps_only=False, base_only=False, config=None):
    """Applies checkpoint weights (FPS-only or mixed LoRA+FPS)."""
    logging.info(f"Loading checkpoint: {checkpoint_path}")
    
    checkpoint_type = detect_checkpoint_type(checkpoint_path)
    logging.info(f"Detected checkpoint type: {checkpoint_type}")
    
    # Verify FPS configuration for fps_only mode
    if fps_only and config:
        verify_fps_config_applied(wan_t2v_pipeline, config)
    
    # Debug: Show FPS parameter values BEFORE loading checkpoint
    logging.info("🔍 FPS parameter values BEFORE checkpoint loading:")
    params_before = {}
    
    # FPS MLP parameters
    for name, param in wan_t2v_pipeline.transformer.named_parameters():
        if 'fps_conditioning' in name:
            param_mean = param.data.mean().item()
            param_std = param.data.std().item() if param.numel() > 1 else 0.0
            param_max = param.data.abs().max().item()
            params_before[name] = (param_mean, param_std, param_max)
            logging.info(f"  {name}: mean={param_mean:.6f}, std={param_std:.6f}, max={param_max:.6f}")
    
    # FPS Adapter parameters (sample first few)
    adapter_count = 0
    for name, param in wan_t2v_pipeline.transformer.named_parameters():
        if 'fps_adapter' in name and adapter_count < 6:  # Show first 6 adapter params
            param_mean = param.data.mean().item()
            param_std = param.data.std().item() if param.numel() > 1 else 0.0
            param_max = param.data.abs().max().item()
            params_before[name] = (param_mean, param_std, param_max)
            
            if 'gate_alpha' in name:
                gate_sigmoid = torch.sigmoid(param.data).item()
                logging.info(f"  {name}: alpha={param_mean:.6f} -> gate={gate_sigmoid:.6f}")
            else:
                logging.info(f"  {name}: mean={param_mean:.6f}, std={param_std:.6f}, max={param_max:.6f}")
    
    # Also check for fixed gate buffers
    for name, buffer in wan_t2v_pipeline.transformer.named_buffers():
        if 'fps_adapter' in name and 'gate_fixed' in name and adapter_count < 6:
            gate_value = buffer.data.item()
            logging.info(f"  {name}: gate={gate_value:.6f} (FIXED MODE)")
            adapter_count += 1
            adapter_count += 1
    
    if adapter_count >= 6:
        logging.info(f"  ... and {sum(1 for n, _ in wan_t2v_pipeline.transformer.named_parameters() if 'fps_adapter' in n) - 6} more adapter parameters")
    
    # Configure base LoRA adapter based on selective loading mode
    configure_base_lora = False
    
    if fps_only:
        logging.info("🎯 FPS-ONLY MODE: Skipping base LoRA configuration")
        logging.info("   ✅ FPS adapters were already configured from TOML during pipeline initialization")
        logging.info("   ✅ FPS configuration from TOML will be used (rank, gate_mode, etc.)")
        configure_base_lora = False
    elif base_only:
        logging.info("🎯 BASE-ONLY MODE: Configuring only base LoRA adapter")
        configure_base_lora = True
    else:
        # Normal mode - configure based on checkpoint type
        if checkpoint_type in ['lora_and_fps', 'unknown']:
            configure_base_lora = True
        elif checkpoint_type == 'fps_only':
            configure_base_lora = False
    
    if configure_base_lora:
        adapter_config = {"type": "lora", "rank": rank, "alpha": rank, "dropout": 0.0, "dtype": dtype}
        wan_t2v_pipeline.configure_adapter(adapter_config)
        logging.info(f"Configured base LoRA adapter with rank {rank}")
    
    # Load weights with selective parameter filtering
    load_adapter_weights_selective(wan_t2v_pipeline, checkpoint_path, fps_only, base_only)
    
    # Debug: Show FPS parameter values AFTER loading checkpoint
    logging.info("✅ FPS parameter values AFTER checkpoint loading:")
    
    # FPS MLP parameters
    for name, param in wan_t2v_pipeline.transformer.named_parameters():
        if 'fps_conditioning' in name:
            param_mean = param.data.mean().item()
            param_std = param.data.std().item() if param.numel() > 1 else 0.0
            param_max = param.data.abs().max().item()
            
            if name in params_before:
                before_mean, before_std, before_max = params_before[name]
                mean_changed = abs(param_mean - before_mean) > 1e-6
                change_indicator = "📝 CHANGED" if mean_changed else "🔒 UNCHANGED"
                logging.info(f"  {name}: mean={param_mean:.6f}, std={param_std:.6f}, max={param_max:.6f} {change_indicator}")
                if mean_changed:
                    logging.info(f"    Mean change: {before_mean:.6f} -> {param_mean:.6f} (Δ={param_mean-before_mean:.6f})")
            else:
                logging.info(f"  {name}: mean={param_mean:.6f}, std={param_std:.6f}, max={param_max:.6f} 🆕 NEW")
    
    # FPS Adapter parameters (sample first few)
    adapter_count = 0
    for name, param in wan_t2v_pipeline.transformer.named_parameters():
        if 'fps_adapter' in name and adapter_count < 6:
            param_mean = param.data.mean().item()
            param_std = param.data.std().item() if param.numel() > 1 else 0.0
            param_max = param.data.abs().max().item()
            
            if name in params_before:
                before_mean, before_std, before_max = params_before[name]
                mean_changed = abs(param_mean - before_mean) > 1e-6
                change_indicator = "📝 CHANGED" if mean_changed else "🔒 UNCHANGED"
                
                if 'gate_alpha' in name:
                    gate_sigmoid = torch.sigmoid(param.data).item()
                    logging.info(f"  {name}: alpha={param_mean:.6f} -> gate={gate_sigmoid:.6f} {change_indicator}")
                else:
                    logging.info(f"  {name}: mean={param_mean:.6f}, std={param_std:.6f}, max={param_max:.6f} {change_indicator}")
                
                if mean_changed:
                    logging.info(f"    Mean change: {before_mean:.6f} -> {param_mean:.6f} (Δ={param_mean-before_mean:.6f})")
            else:
                if 'gate_alpha' in name:
                    gate_sigmoid = torch.sigmoid(param.data).item()
                    logging.info(f"  {name}: alpha={param_mean:.6f} -> gate={gate_sigmoid:.6f} 🆕 NEW")
                else:
                    logging.info(f"  {name}: mean={param_mean:.6f}, std={param_std:.6f}, max={param_max:.6f} 🆕 NEW")
            adapter_count += 1
    
    # Also check for fixed gate buffers after loading
    logging.info("✅ Fixed gate buffer values AFTER checkpoint loading:")
    for name, buffer in wan_t2v_pipeline.transformer.named_buffers():
        if 'fps_adapter' in name and 'gate_fixed' in name:
            gate_value = buffer.data.item()
            logging.info(f"  {name}: gate={gate_value:.6f} (FIXED MODE)")
    
    # Verify loaded parameters
    fps_mlp_params = 0
    fps_adapter_params = 0
    base_lora_params = 0
    
    for name, param in wan_t2v_pipeline.transformer.named_parameters():
        if 'fps_conditioning' in name:
            fps_mlp_params += 1
        elif 'fps_adapter' in name:
            fps_adapter_params += 1
        elif 'lora' in name.lower() and 'fps' not in name:
            base_lora_params += 1
    
    logging.info(f"✅ Checkpoint loaded successfully!")
    logging.info(f"  FPS MLP parameters: {fps_mlp_params}")
    logging.info(f"  FPS Adapter parameters: {fps_adapter_params}")
    logging.info(f"  Base LoRA parameters: {base_lora_params}")
    
    if fps_mlp_params == 0 and fps_adapter_params == 0:
        logging.warning("⚠️  No FPS parameters found!")
    
    return wan_t2v_pipeline

def generate_video_with_fps(pipeline, config, prompt, n_prompt, fps=60, seed=42, steps=25, scale=7.0, frames=49, size=(512, 320), shift=3.0, force_gate_one=False):
    """Generates a video with specific FPS conditioning."""
    logging.info(f"🎬 Generating video with FPS={fps}")
    if force_gate_one:
        logging.info("🔧 DIAGNOSTIC MODE: Will force gates to 1.0")
    logging.info(f"  Prompt: {prompt}")
    logging.info(f"  Steps: {steps}, Frames: {frames}, Size: {size}")
    
    device = pipeline.transformer.device
    
    # Set seed for reproducibility
    torch.manual_seed(seed)
    
    # Encode text
    text_encoder_fn = pipeline.get_call_text_encoder_fn(pipeline.text_encoder.model.to(device))
    cond_inputs = text_encoder_fn([prompt], is_video=True)
    uncond_inputs = text_encoder_fn([n_prompt], is_video=True)
    
    # Prepare FPS tensor
    # fps can be a scalar (single-condition) or list (multi-condition)
    if isinstance(fps, list):
        # Multi-condition: fps is already a list like [0.5, 0.02]
        fps_values = torch.tensor([fps], dtype=torch.float32, device=device)  # Shape: [1, num_conditions]
        logging.info(f"  FPS tensor (multi-condition): {fps_values.tolist()}")
    else:
        # Single-condition: fps is a scalar like 12.0
        fps_values = torch.tensor([fps], dtype=torch.float32, device=device)  # Shape: [1]
        logging.info(f"  FPS tensor (single-condition): {fps_values.tolist()}")
    
    # 🔍 DEBUG: Check FPS parameters before generation
    logging.info("🔍 FPS parameter status before generation:")
    fps_param_count = 0
    fps_condition_count = 0
    fps_adapter_count = 0
    
    for name, param in pipeline.transformer.named_parameters():
        if 'fps_conditioning' in name:
            fps_condition_count += 1
            if fps_condition_count <= 2:  # Show first 2 FPS conditioning params
                param_mean = param.data.mean().item()
                param_max = param.data.abs().max().item()
                logging.info(f"  {name}: mean={param_mean:.6f}, max={param_max:.6f}")
        elif 'fps_adapter' in name:
            fps_adapter_count += 1
            if 'gate_alpha' in name and fps_adapter_count <= 3:  # Show first 3 gate values
                gate_alpha = param.data.item()
                gate_sigmoid = torch.sigmoid(param.data).item()
                logging.info(f"  {name}: alpha={gate_alpha:.6f} -> gate={gate_sigmoid:.6f}")
    
    # Also check for fixed gate buffers
    fixed_gate_count = 0
    for name, buffer in pipeline.transformer.named_buffers():
        if 'fps_adapter' in name and 'gate_fixed' in name and fixed_gate_count <= 3:
            gate_value = buffer.data.item()
            logging.info(f"  {name}: gate={gate_value:.6f} (FIXED MODE)")
            fixed_gate_count += 1
    
    logging.info(f"  Found {fps_condition_count} FPS conditioning params, {fps_adapter_count} FPS adapter params")
    
    # 🔧 DIAGNOSTIC: Force all FPS adapter gates to 1.0 for maximum impact
    if force_gate_one:
        logging.info("🔧 DIAGNOSTIC MODE: Forcing all FPS adapter gates to 1.0 for maximum FPS impact")
        gates_forced = 0
        
        # Handle learned gates (gate_alpha parameters)
        for name, param in pipeline.transformer.named_parameters():
            if 'fps_adapter' in name and 'gate_alpha' in name:
                # Set gate_alpha to a very large positive value to force sigmoid -> 1.0
                # sigmoid(10) ≈ 0.99995, which is effectively 1.0
                with torch.no_grad():
                    param.data.fill_(10.0)
                gates_forced += 1
        
        # Handle fixed gates (gate_fixed buffers)
        for name, buffer in pipeline.transformer.named_buffers():
            if 'fps_adapter' in name and 'gate_fixed' in name:
                # Set fixed gate buffer to 1.0
                with torch.no_grad():
                    buffer.data.fill_(1.0)
                gates_forced += 1
        
        logging.info(f"  Forced {gates_forced} gate values to maximum (1.0)")
        
        # Verify the forced gate values
        logging.info("  Verification - Gate values after forcing:")
        # Check learned gates
        for name, param in pipeline.transformer.named_parameters():
            if 'fps_adapter' in name and 'gate_alpha' in name:
                gate_alpha = param.data.item()
                gate_sigmoid = torch.sigmoid(param.data).item()
                logging.info(f"    {name}: alpha={gate_alpha:.1f} -> gate={gate_sigmoid:.6f} (LEARNED)")
                break  # Just show one example
        
        # Check fixed gates
        for name, buffer in pipeline.transformer.named_buffers():
            if 'fps_adapter' in name and 'gate_fixed' in name:
                gate_value = buffer.data.item()
                logging.info(f"    {name}: gate={gate_value:.6f} (FIXED)")
                break  # Just show one example
    
    # Initialize latents
    vae_stride = [4, 8, 8]
    target_shape = (16, frames // vae_stride[0], size[1] // vae_stride[1], size[0] // vae_stride[2])
    latents = torch.randn(target_shape, device=device)
    
    layers = pipeline.to_layers()
    initial_layer, transformer_layers, final_layer = layers[0], layers[1:-1], layers[-1]
    
    # Setup scheduler
    scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
    scheduler.set_timesteps(steps, device=device, shift=shift)
    timesteps = scheduler.timesteps
    
    # Denoising loop
    for step_idx, t in enumerate(tqdm(timesteps, desc=f"FPS={fps}")):
        t_batch = torch.full((1,), t, device=device)
        
        # SUBTASK 4: Fix autocast dtype to match training dtype (already converted to torch.dtype)
        autocast_dtype = config['model']['dtype']  # Already converted by DTYPE_MAP
        
        with torch.no_grad(), torch.autocast('cuda', dtype=autocast_dtype):
            # Unconditional pass
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
            
            # Conditional pass
            initial_input_cond = (
                latents.unsqueeze(0), None, t_batch,
                cond_inputs['text_embeddings'].to(device), 
                cond_inputs['seq_lens'].to(device), 
                None, fps_values
            )
            
            layer_outputs = initial_layer(initial_input_cond)
            x, e, e0, seq_lens, grid_sizes, freqs, context, fps_conditioning = layer_outputs
            
            # 🔍 DEBUG: Check fps_conditioning tensor values (only for first step)
            if step_idx == 0:
                fps_mean = fps_conditioning.mean().item()
                fps_std = fps_conditioning.std().item()
                fps_max = fps_conditioning.abs().max().item()
                logging.info(f"  [Step {step_idx}] FPS conditioning tensor: mean={fps_mean:.6f}, std={fps_std:.6f}, max={fps_max:.6f}")
            
            for transformer_layer in transformer_layers:
                layer_outputs = transformer_layer((x, e, e0, seq_lens, grid_sizes, freqs, context, fps_conditioning))
                x, e, e0, seq_lens, grid_sizes, freqs, context, fps_conditioning = layer_outputs
                
            noise_pred_cond = final_layer((x, e, e0, seq_lens, grid_sizes, freqs, context, fps_conditioning))
            
            # CFG
            noise_pred = noise_pred_uncond.squeeze(0) + scale * (noise_pred_cond.squeeze(0) - noise_pred_uncond.squeeze(0))
        
        latents = scheduler.step(noise_pred, t, latents.unsqueeze(0), return_dict=False)[0].squeeze(0)
    
    # Decode to video
    logging.info("Decoding latents...")
    video_tensor = pipeline.vae.decode([latents])[0]
    
    # 🔍 DEBUG: Final verification of FPS parameters after generation
    logging.info("🔍 FPS parameter status after generation:")
    for name, param in pipeline.transformer.named_parameters():
        if 'fps_conditioning' in name and 'lin1.weight' in name:  # Check one representative param
            param_mean = param.data.mean().item()
            param_max = param.data.abs().max().item()
            logging.info(f"  {name}: mean={param_mean:.6f}, max={param_max:.6f}")
            break
    
    logging.info(f"✅ Generation complete for FPS={fps}")
    
    return video_tensor

def save_video_result(tensor, fps, prompt_short, output_dir, size):
    """Saves video with descriptive filename."""
    if tensor is None:
        logging.warning("Video tensor is None, skipping save.")
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_prompt = "".join(c for c in prompt_short if c.isalnum() or c in (' ', '_')).strip()[:30]

    # Handle both scalar and list fps values for filename
    if isinstance(fps, list):
        # Multi-condition: join with underscore, e.g., [0.5, 0.02] -> "0_50_0_02"
        fps_str = '_'.join(f"{v:.2f}" for v in fps).replace('.', '_')
    else:
        # Single-condition: e.g., 12.0 -> "12_00"
        fps_str = f"{fps:.2f}".replace('.', '_')

    filename = f"fps_{fps_str}_{clean_prompt.replace(' ', '_')}_{size[0]}x{size[1]}_{timestamp}.mp4"
    filepath = os.path.join(output_dir, filename)
    
    logging.info(f"💾 Saving: {filename}")
    cache_video(tensor=tensor[None], save_file=filepath, fps=16, normalize=True, value_range=(-1, 1))
    
    return filepath

def run_fps_experiments(pipeline, config, base_prompt, n_prompt, fps_values, output_dir, force_gate_one=False, **generation_kwargs):
    """Runs multiple FPS conditioning experiments."""
    logging.info("🧪 Starting FPS conditioning experiments...")
    logging.info(f"  FPS values to test: {fps_values}")
    logging.info(f"  Base prompt: {base_prompt}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    for i, fps in enumerate(fps_values, 1):
        logging.info(f"\n--- Experiment {i}/{len(fps_values)}: FPS = {fps} ---")
        
        try:
            # Generate video
            video_tensor = generate_video_with_fps(
                pipeline=pipeline,
                config=config,  # Pass config for proper dtype handling
                prompt=base_prompt,
                n_prompt=n_prompt,
                fps=fps,
                force_gate_one=force_gate_one,
                **generation_kwargs
            )
            
            # Save result
            output_file = save_video_result(
                tensor=video_tensor,
                fps=fps,
                prompt_short=base_prompt.split()[0:3],  # First few words
                output_dir=output_dir,
                size=generation_kwargs.get('size', (512, 320))
            )
            
            # Collect stats for comparison
            stats = {
                'fps': fps,
                'file': output_file,
                'mean': video_tensor.mean().item(),
                'std': video_tensor.std().item(),
                'min': video_tensor.min().item(),
                'max': video_tensor.max().item()
            }
            results.append(stats)
            
            logging.info(f"✅ Experiment {i} complete")
            
        except Exception as e:
            logging.error(f"❌ Experiment {i} failed: {e}")
            results.append({'fps': fps, 'error': str(e)})
    
    # Analysis
    logging.info(f"\n📊 FPS Conditioning Analysis:")
    logging.info(f"{'FPS':<15} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10} {'File':<50}")
    logging.info("-" * 100)
    
    successful_results = [r for r in results if 'error' not in r]
    for result in results:
        # Format FPS value (handle both scalar and list)
        fps_str = str(result['fps']) if isinstance(result['fps'], list) else f"{result['fps']}"

        if 'error' in result:
            logging.info(f"{fps_str:<15} ERROR: {result['error']}")
        else:
            logging.info(f"{fps_str:<15} {result['mean']:<10.4f} {result['std']:<10.4f} "
                        f"{result['min']:<10.4f} {result['max']:<10.4f} {os.path.basename(result['file'])}")
    
    # Check if FPS conditioning is working
    if len(successful_results) >= 2:
        means = [r['mean'] for r in successful_results]
        mean_variation = max(means) - min(means)
        if mean_variation > 0.01:  # Threshold for meaningful difference
            logging.info(f"\n✅ SUCCESS: FPS conditioning is working! Mean variation: {mean_variation:.4f}")
        else:
            logging.info(f"\n⚠️  WARNING: Low variation between FPS values. Mean variation: {mean_variation:.4f}")
    
    return results

def validate_fps_config(config):
    """Validates that the config has FPS parameter definitions for selective loading."""
    model_config = config.get('model', {})
    
    # Check for required FPS parameters
    fps_params = [
        'fps_adapter_rank', 'fps_adapter_num_tokens', 'fps_embed_dim', 
        'fps_lora_alpha', 'fps_condition_blocks'
    ]
    
    missing_params = [param for param in fps_params if param not in model_config]
    if missing_params:
        raise ValueError(
            f"--fps_only requires FPS parameters in config, but missing: {missing_params}. "
            f"Cannot load FPS-only parameters from checkpoint."
        )
    
    logging.info("✅ Config validation passed: FPS parameters found in configuration")

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
    
    logging.info("✅ Config validation passed: Base LoRA parameters found in configuration")

def load_prompts_from_folder(folder_path):
    """
    Load prompts from text files in a folder.

    Args:
        folder_path: Path to folder containing .txt files with prompts

    Returns:
        List of (prompt_id, prompt_text) tuples
    """
    from pathlib import Path

    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"Prompt folder does not exist or is not a directory: {folder_path}")

    prompts = []
    txt_files = sorted(folder.glob('*.txt'))

    if not txt_files:
        raise ValueError(f"No .txt files found in prompt folder: {folder_path}")

    logging.info(f"\n📝 Loading prompts from folder: {folder_path}")
    for txt_file in txt_files:
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                prompt_text = f.read().strip()
                if prompt_text:  # Skip empty files
                    prompts.append((txt_file.stem, prompt_text))
                    logging.info(f"  ✓ Loaded '{txt_file.stem}': {prompt_text[:60]}...")
        except Exception as e:
            logging.warning(f"  ✗ Failed to read {txt_file.name}: {e}")

    if not prompts:
        raise ValueError(f"No valid prompts loaded from folder: {folder_path}")

    logging.info(f"📝 Total: {len(prompts)} prompts loaded\n")
    return prompts


def parse_fps_values(fps_values_flat, num_conditions):
    """
    Parse flat fps_values list into groups based on num_conditions.

    Args:
        fps_values_flat: Flat list of floats from command line
        num_conditions: Number of conditions (1 for single, 2+ for multi)

    Returns:
        List of fps_values, where each element is:
        - Single-condition: scalar float
        - Multi-condition: list of floats

    Examples:
        Single-condition (num_conditions=1):
            Input: [12.0, 24.0, 60.0]
            Output: [12.0, 24.0, 60.0]

        Multi-condition (num_conditions=2):
            Input: [0.5, 0.02, 1.0, 0.05, 0.125, 0.1]
            Output: [[0.5, 0.02], [1.0, 0.05], [0.125, 0.1]]
    """
    if num_conditions == 1:
        # Single condition: each value is a separate experiment
        return fps_values_flat
    else:
        # Multi-condition: group values into tuples of size num_conditions
        if len(fps_values_flat) % num_conditions != 0:
            raise ValueError(
                f"fps_values length ({len(fps_values_flat)}) must be divisible by "
                f"num_conditions ({num_conditions}). Got {len(fps_values_flat)} values "
                f"which doesn't divide evenly into groups of {num_conditions}."
            )

        grouped = []
        for i in range(0, len(fps_values_flat), num_conditions):
            group = fps_values_flat[i:i+num_conditions]
            grouped.append(group)

        return grouped


def main():
    parser = argparse.ArgumentParser(description='Multi-FPS experiment script - TOML-aligned')
    parser.add_argument('--config', required=True, help='Path to TOML configuration file')
    parser.add_argument('--checkpoint', required=False, help='Path to FPS checkpoint (not needed if using --clean)')
    parser.add_argument('--output_dir', default='./fps_experiments_align', help='Output directory')
    parser.add_argument('--fps_values', nargs='+', type=float, default=[12, 24, 60],
                        help='FPS values to test. ' +
                             'Single-condition: --fps_values 12 24 60 (3 experiments). ' +
                             'Multi-condition (2): --fps_values 0.5 0.02  1.0 0.05  0.125 0.1 ' +
                             '(3 experiments with 2 conditions each, values are grouped automatically)')
    parser.add_argument('--prompt', default=None, help='Generation prompt (single string)')
    parser.add_argument('--prompt_folder', default=None, help='Folder containing .txt files with prompts (alternative to --prompt)')
    parser.add_argument('--negative_prompt', default='', help='Negative prompt (optional)')
    parser.add_argument('--steps', type=int, default=15, help='Denoising steps')
    parser.add_argument('--frames', type=int, default=33, help='Number of frames')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--port', default='29501', help='DeepSpeed port')
    parser.add_argument('--width', type=int, help='Video width (overrides TOML if specified)')
    parser.add_argument('--height', type=int, help='Video height (overrides TOML if specified)')
    parser.add_argument('--scale', type=float, help='CFG scale (overrides TOML if specified)')
    parser.add_argument('--force_gate_one', action='store_true', help='🔧 DIAGNOSTIC: Force all FPS adapter gates to 1.0 for maximum FPS impact')
    parser.add_argument('--fps_only', action='store_true', help='Load only FPS-related parameters from checkpoint (ignore base LoRA)')
    parser.add_argument('--base_only', action='store_true', help='Load only base LoRA parameters from checkpoint (ignore FPS parameters)')
    parser.add_argument('--clean', action='store_true', help='Use clean backbone without loading any LoRA (neither base LoRA nor FPS adapters)')

    args = parser.parse_args()

    # Validate prompt arguments
    if args.prompt is None and args.prompt_folder is None:
        parser.error("Either --prompt or --prompt_folder must be provided")
    if args.prompt is not None and args.prompt_folder is not None:
        parser.error("--prompt and --prompt_folder are mutually exclusive. Choose one.")

    # Validate selective loading arguments
    exclusive_modes = [args.fps_only, args.base_only, args.clean]
    if sum(exclusive_modes) > 1:
        parser.error("--fps_only, --base_only, and --clean are mutually exclusive. Choose at most one.")

    # Validate checkpoint requirement
    if not args.clean and not args.checkpoint:
        parser.error("--checkpoint is required unless using --clean mode")

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
        if num_conditions > 1:
            logging.info(f"   → Values will be grouped into sets of {num_conditions}")

        # Parse fps_values based on num_conditions
        fps_values_parsed = parse_fps_values(args.fps_values, num_conditions)
        logging.info(f"📊 Parsed FPS values from: {args.fps_values}")
        if num_conditions == 1:
            logging.info(f"  Single-condition: {fps_values_parsed}")
        else:
            logging.info(f"  Multi-condition ({num_conditions} conditions per experiment):")
            for i, vals in enumerate(fps_values_parsed, 1):
                logging.info(f"    Experiment {i}: [{', '.join(str(v) for v in vals)}]")

        # Validate selective loading requirements and apply checkpoint
        if args.clean:
            logging.info("🧹 CLEAN MODE: Using original backbone without any LoRA")
            logging.info("   → Skipping checkpoint loading entirely")
            logging.info("   → No base LoRA will be applied")
            logging.info("   → No FPS adapters will be applied")
        else:
            if args.fps_only:
                validate_fps_config(config)
                logging.info("🎯 FPS-ONLY MODE: Will load only FPS-related parameters")
            elif args.base_only:
                validate_base_lora_config(config)
                logging.info("🎯 BASE-ONLY MODE: Will load only base LoRA parameters")

            # Apply checkpoint
            logging.info("Applying checkpoint...")
            # Get base LoRA rank from TOML config or default
            base_rank = config.get('adapter', {}).get('rank', 32)
            pipeline = apply_checkpoint(pipeline, args.checkpoint, rank=base_rank,
                                       fps_only=args.fps_only, base_only=args.base_only, config=config)
        
        # SUBTASK 2: Use TOML parameters for inference settings (with CLI overrides)
        # Extract inference parameters from TOML config or use defaults
        width = args.width if args.width is not None else 512
        height = args.height if args.height is not None else 512
        scale = args.scale if args.scale is not None else 6.0
        
        logging.info("Inference parameters:")
        logging.info(f"  Resolution: {width}x{height}")
        logging.info(f"  CFG scale: {scale}")
        logging.info(f"  Steps: {args.steps}")
        logging.info(f"  Frames: {args.frames}")
        logging.info(f"  Negative prompt: '{args.negative_prompt}'")

        # Prepare prompts list
        if args.prompt_folder:
            # Load multiple prompts from folder
            prompts = load_prompts_from_folder(args.prompt_folder)
        else:
            # Single prompt from command line
            prompts = [("single", args.prompt)]
            logging.info(f"  Prompt: '{args.prompt}'")

        # Run experiments for each prompt
        all_results = []
        for prompt_idx, (prompt_id, prompt_text) in enumerate(prompts, 1):
            if len(prompts) > 1:
                logging.info(f"\n{'='*80}")
                logging.info(f"PROCESSING PROMPT {prompt_idx}/{len(prompts)}: {prompt_id}")
                logging.info(f"{'='*80}")
                logging.info(f"Prompt text: {prompt_text}")

            # Create subdirectory for this prompt if using multiple prompts
            if len(prompts) > 1:
                prompt_output_dir = os.path.join(args.output_dir, prompt_id)
            else:
                prompt_output_dir = args.output_dir

            # Run experiments for this prompt
            results = run_fps_experiments(
                pipeline=pipeline,
                config=config,  # Pass TOML config for proper dtype handling
                base_prompt=prompt_text,
                n_prompt=args.negative_prompt,  # Use provided negative prompt or empty string
                fps_values=fps_values_parsed,  # Use parsed fps_values (handles multi-condition)
                output_dir=prompt_output_dir,
                seed=args.seed,
                steps=args.steps,
                frames=args.frames,
                size=(width, height),
                scale=scale,
                shift=3.0,
                force_gate_one=args.force_gate_one
            )

            all_results.append({
                'prompt_id': prompt_id,
                'prompt_text': prompt_text,
                'results': results,
                'output_dir': prompt_output_dir
            })

            logging.info(f"✅ Prompt '{prompt_id}' complete!")

        # Final summary
        logging.info(f"\n{'='*80}")
        logging.info(f"🎉 ALL EXPERIMENTS COMPLETE!")
        logging.info(f"{'='*80}")
        logging.info(f"Total prompts processed: {len(prompts)}")
        logging.info(f"Total videos generated: {sum(len(r['results']) for r in all_results)}")
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