import argparse
import os
import wandb
from datetime import datetime, timezone
import shutil
import glob
import time
import random
import json
import inspect
from pathlib import Path
from collections import defaultdict

import toml
import deepspeed
from deepspeed import comm as dist
from deepspeed.runtime.pipe import module as ds_pipe_module
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import multiprocess as mp
import numpy as np

from utils import dataset as dataset_util
from utils import common
from utils.common import is_main_process, get_rank, DTYPE_MAP, empty_cuda_cache
import utils.saver
from utils.isolate_rng import isolate_rng
from utils.patches import apply_patches
from utils.unsloth_utils import unsloth_checkpoint
from utils.pipeline import ManualPipelineModule

# needed for broadcasting Queue in dataset.py
mp.current_process().authkey = b'afsaskgfdjh4'

wandb_enable = False

TIMESTEP_QUANTILES_FOR_EVAL = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

parser = argparse.ArgumentParser()
parser.add_argument('--config', help='Path to TOML configuration file.')
parser.add_argument('--local_rank', type=int, default=-1,
                    help='local rank passed from distributed launcher')
parser.add_argument('--resume_from_checkpoint', nargs='?', const=True, default=None,
                    help='resume training from checkpoint. If no value is provided, resume from the most recent checkpoint. If a folder name is provided, resume from that specific folder.')
parser.add_argument('--regenerate_cache', action='store_true', help='Force regenerate cache.')
parser.add_argument('--cache_only', action='store_true', help='Cache model inputs then exit.')
parser.add_argument('--trust_cache', action='store_true', help='Load from metadata cache files if they exist, without checking if any fingerprints have changed. Can make loading much faster for large datasets.')
parser.add_argument('--i_know_what_i_am_doing', action='store_true', help="Skip certain checks and overrides. You may end up using settings that won't work.")
parser.add_argument('--master_port', type=int, default=29500, help='Master port for distributed training')
parser.add_argument('--dump_dataset', type=Path, default=None, help='Decode cached latents and dump the dataset to this directory.')
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()


class DummyOptimizer(torch.optim.Optimizer):
    def __init__(self):
        self.state = defaultdict(dict)
        self.param_groups = []

    def step(self, closure=None):
        pass

    def zero_grad(self, set_to_none: bool = True):
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass


# Monkeypatch this so it counts all layer parameters, not just trainable parameters.
# This helps it divide the layers between GPUs more evenly when training a LoRA.
def _count_all_layer_params(self):
    param_counts = [0] * len(self._layer_specs)
    for idx, layer in enumerate(self._layer_specs):
        if isinstance(layer, ds_pipe_module.LayerSpec):
            l = layer.build()
            param_counts[idx] = sum(p.numel() for p in l.parameters())
        elif isinstance(layer, nn.Module):
            param_counts[idx] = sum(p.numel() for p in layer.parameters())
    return param_counts
ds_pipe_module.PipelineModule._count_layer_params = _count_all_layer_params


def set_config_defaults(config):
    # Force the user to set this. If we made it a default of 1, it might use a lot of disk space.
    assert 'save_every_n_epochs' in config or 'save_every_n_steps' in config

    config.setdefault('pipeline_stages', 1)
    config.setdefault('activation_checkpointing', False)
    config.setdefault('reentrant_activation_checkpointing', False)
    if config['activation_checkpointing'] == 'unsloth':
        config['reentrant_activation_checkpointing'] = True
    config.setdefault('warmup_steps', 0)
    if 'save_dtype' in config:
        config['save_dtype'] = DTYPE_MAP[config['save_dtype']]

    model_config = config['model']
    model_dtype_str = model_config['dtype']
    model_config['dtype'] = DTYPE_MAP[model_dtype_str]
    if transformer_dtype := model_config.get('transformer_dtype', None):
        model_config['transformer_dtype'] = DTYPE_MAP.get(transformer_dtype, transformer_dtype)
    model_config.setdefault('guidance', 1.0)

    if 'adapter' in config:
        adapter_config = config['adapter']
        adapter_type = adapter_config['type']
        if adapter_config['type'] == 'lora':
            if 'alpha' in adapter_config:
                raise NotImplementedError(
                    'This script forces alpha=rank to make the saved LoRA format simpler and more predictable with downstream inference programs. Please remove alpha from the config.'
                )
            adapter_config['alpha'] = adapter_config['rank']
            adapter_config.setdefault('dropout', 0.0)
            adapter_config.setdefault('dtype', model_dtype_str)
            adapter_config['dtype'] = DTYPE_MAP[adapter_config['dtype']]
        else:
            raise NotImplementedError(f'Adapter type {adapter_type} is not implemented')

    config.setdefault('logging_steps', 1)
    config.setdefault('eval_datasets', [])
    config.setdefault('eval_gradient_accumulation_steps', 1)
    config.setdefault('eval_every_n_steps', None)
    config.setdefault('eval_every_n_epochs', None)
    config.setdefault('eval_before_first_step', True)
    config.setdefault('compile', False)


def get_most_recent_run_dir(output_dir):
    return list(sorted(glob.glob(os.path.join(output_dir, '*'))))[-1]


def print_model_info(model):
    if not is_main_process():
        return
    print(model)
    for name, module in model.named_modules():
        print(f'{type(module)}: {name}')
        for pname, p in module.named_parameters(recurse=False):
            print(pname)
            print(p.dtype)
            print(p.device)
            print(p.requires_grad)
            print()


# Need to preload all micro batches since pulling from the dataloader does IPC between the
# first and last stage. Can't do that during the train or inference pipeline schedule execution
# because it conflicts with the send / recv steps.
def get_data_iterator_for_step(dataloader, engine, num_micro_batches=None):
    num_micro_batches = num_micro_batches or engine.micro_batches
    if not (engine.is_first_stage() or engine.is_last_stage()):
        return None
    dataloader_iter = iter(dataloader)
    items = [next(dataloader_iter) for _ in range(num_micro_batches)]
    return iter(items)


def evaluate_single(model_engine, eval_dataloader, eval_gradient_accumulation_steps, quantile, pbar=None):
    eval_dataloader.set_eval_quantile(quantile)
    total_loss = 0
    count = 0
    while True:
        model_engine.reset_activation_shape()
        iterator = get_data_iterator_for_step(eval_dataloader, model_engine, num_micro_batches=eval_gradient_accumulation_steps)
        loss = model_engine.eval_batch(iterator, num_micro_batches=eval_gradient_accumulation_steps).item()
        eval_dataloader.sync_epoch()
        if pbar:
            pbar.update(1)
        total_loss += loss
        count += 1
        if eval_dataloader.epoch == 2:
            break

    eval_dataloader.reset()
    return total_loss / count


def _evaluate(model_engine, eval_dataloaders, tb_writer, step, eval_gradient_accumulation_steps):
    pbar_total = 0
    for eval_dataloader in eval_dataloaders.values():
        pbar_total += len(eval_dataloader) * len(TIMESTEP_QUANTILES_FOR_EVAL) // eval_gradient_accumulation_steps
    if is_main_process():
        print('Running eval')
        pbar = tqdm(total=pbar_total)
    else:
        pbar = None

    start = time.time()
    for name, eval_dataloader in eval_dataloaders.items():
        losses = []
        for quantile in TIMESTEP_QUANTILES_FOR_EVAL:
            loss = evaluate_single(model_engine, eval_dataloader, eval_gradient_accumulation_steps, quantile, pbar=pbar)
            losses.append(loss)
            if is_main_process():
                tb_writer.add_scalar(f'{name}/loss_quantile_{quantile:.2f}', loss, step)
                if wandb_enable:
                    wandb.log({f'{name}/loss_quantile_{quantile:.2f}': loss, 'step': step})
        avg_loss = sum(losses) / len(losses)
        if is_main_process():
            tb_writer.add_scalar(f'{name}/loss', avg_loss, step)
            if wandb_enable:
                wandb.log({f'{name}/loss': avg_loss, 'step': step})

    duration = time.time() - start
    if is_main_process():
        tb_writer.add_scalar('eval/eval_time_sec', duration, step)
        if wandb_enable:
            wandb.log({'eval/eval_time_sec': duration, 'step': step})
        pbar.close()


def evaluate(model, model_engine, eval_dataloaders, tb_writer, step, eval_gradient_accumulation_steps, disable_block_swap):
    if len(eval_dataloaders) == 0:
        return
    empty_cuda_cache()
    model.prepare_block_swap_inference(disable_block_swap=disable_block_swap)
    with torch.no_grad(), isolate_rng():
        seed = get_rank()
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        _evaluate(model_engine, eval_dataloaders, tb_writer, step, eval_gradient_accumulation_steps)
    empty_cuda_cache()
    model.prepare_block_swap_training()


def distributed_init(args):
    """Initialize distributed training environment."""
    world_size = int(os.getenv('WORLD_SIZE', '1'))
    rank = int(os.getenv('RANK', '0'))
    local_rank = args.local_rank

    # Set environment variables for distributed training
    os.environ['MASTER_ADDR'] = os.getenv('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = str(args.master_port)

    return world_size, rank, local_rank


def get_prodigy_d(optimizer):
    d = 0
    for group in optimizer.param_groups:
        d += group['d']
    return d / len(optimizer.param_groups)


def _get_automagic_lrs(optimizer):
    lrs = []
    for group in optimizer.param_groups:
        for p in group['params']:
            state = optimizer.state[p]
            lr = optimizer._get_lr(group, state)
            lrs.append(lr)
    lrs = torch.stack(lrs)
    return lrs, lrs.mean()


if __name__ == '__main__':
    apply_patches()

    with open(args.config) as f:
        # Inline TOML tables are not pickleable, which messes up the multiprocessing dataset stuff. This is a workaround.
        config = json.loads(json.dumps(toml.load(f)))

    set_config_defaults(config)
    common.AUTOCAST_DTYPE = config['model']['dtype']
    dataset_util.UNCOND_FRACTION = config.get('uncond_fraction', 0.0)
    if map_num_proc := config.get('map_num_proc', None):
        dataset_util.NUM_PROC = map_num_proc

    # Initialize distributed environment before deepspeed
    world_size, rank, local_rank = distributed_init(args)

    # Now initialize deepspeed
    deepspeed.init_distributed()

    # needed for broadcasting Queue in dataset.py
    torch.cuda.set_device(dist.get_rank())

    resume_from_checkpoint = (
        args.resume_from_checkpoint if args.resume_from_checkpoint is not None
        else config.get('resume_from_checkpoint', False)
    )
    regenerate_cache = (
        args.regenerate_cache if args.regenerate_cache is not None
        else config.get('regenerate_cache', False)
    )

    model_type = config['model']['type']

    if model_type == 'flux':
        from models import flux
        model = flux.FluxPipeline(config)
    elif model_type == 'ltx-video':
        from models import ltx_video
        model = ltx_video.LTXVideoPipeline(config)
    elif model_type == 'hunyuan-video':
        from models import hunyuan_video
        model = hunyuan_video.HunyuanVideoPipeline(config)
    elif model_type == 'sdxl':
        from models import sdxl
        model = sdxl.SDXLPipeline(config)
    elif model_type == 'cosmos':
        from models import cosmos
        model = cosmos.CosmosPipeline(config)
    elif model_type == 'lumina_2':
        from models import lumina_2
        model = lumina_2.Lumina2Pipeline(config)
    elif model_type == 'wan':
        from models.wan import wan
        model = wan.WanPipeline(config)
    elif model_type == 'chroma':
        from models import chroma
        model = chroma.ChromaPipeline(config)
    elif model_type == 'hidream':
        from models import hidream
        model = hidream.HiDreamPipeline(config)
    elif model_type == 'sd3':
        from models import sd3
        model = sd3.SD3Pipeline(config)
    elif model_type == 'cosmos_predict2':
        from models import cosmos_predict2
        model = cosmos_predict2.CosmosPredict2Pipeline(config)
    elif model_type == 'omnigen2':
        from models import omnigen2
        model = omnigen2.OmniGen2Pipeline(config)
    elif model_type == 'qwen_image':
        from models import qwen_image
        model = qwen_image.QwenImagePipeline(config)
    else:
        raise NotImplementedError(f'Model type {model_type} is not implemented')

    # import sys, PIL
    # test_image = sys.argv[1]
    # with torch.no_grad():
    #     vae = model.get_vae().to('cuda')
    #     latents = dataset.encode_pil_to_latents(PIL.Image.open(test_image), vae)
    #     pil_image = dataset.decode_latents_to_pil(latents, vae)
    #     pil_image.save('test.jpg')
    # quit()

    with open(config['dataset']) as f:
        dataset_config = toml.load(f)
    gradient_release = config['optimizer'].get('gradient_release', False)
    ds_config = {
        'train_micro_batch_size_per_gpu': config.get('micro_batch_size_per_gpu', 1),
        'gradient_accumulation_steps': config.get('gradient_accumulation_steps', 1),
        # Can't do gradient clipping with gradient release, since there are no grads at the end of the step anymore.
        'gradient_clipping': 0. if gradient_release else config.get('gradient_clipping', 1.0),
        'steps_per_print': config.get('steps_per_print', 1),
    }
    caching_batch_size = config.get('caching_batch_size', 1)
    dataset_manager = dataset_util.DatasetManager(model, regenerate_cache=regenerate_cache, trust_cache=args.trust_cache, caching_batch_size=caching_batch_size)

    train_data = dataset_util.Dataset(dataset_config, model, skip_dataset_validation=args.i_know_what_i_am_doing)
    dataset_manager.register(train_data)

    eval_data_map = {}
    for i, eval_dataset in enumerate(config['eval_datasets']):
        if type(eval_dataset) == str:
            name = f'eval{i}'
            config_path = eval_dataset
        else:
            name = eval_dataset['name']
            config_path = eval_dataset['config']
        with open(config_path) as f:
            eval_dataset_config = toml.load(f)
        eval_data_map[name] = dataset_util.Dataset(eval_dataset_config, model, skip_dataset_validation=args.i_know_what_i_am_doing)
        dataset_manager.register(eval_data_map[name])

    # For testing

    # import imageio
    # from pathlib import Path
    # import torch.nn.functional as F
    # dataset_manager.cache(unload_models=False)
    # output_dir = Path('/home/anon/tmp')
    # train_data.post_init(
    #     0,
    #     1,
    #     1,
    #     1,
    # )
    # vae = model.vae
    # vae.model.to('cuda')
    # count = 1
    # for item in train_data:
    #     latents = item['latents'].to('cuda')
    #     h, w = latents.shape[-2:]
    #     mask = item['mask'].to('cuda')
    #     caption = item['caption'][0]
    #     mask = mask.unsqueeze(1)  # make mask (bs, 1, img_h, img_w)
    #     mask = F.interpolate(mask, size=(h, w), mode='nearest-exact')  # resize to latent spatial dimension
    #     mask = mask.unsqueeze(2)  # make mask same number of dims as target
    #     latents = latents * mask.to(latents.device)
    #     video = vae.model.decode(latents, vae.scale).float().clamp_(-1, 1).squeeze(0)
    #     video = torch.permute(video, (1, 2, 3, 0))
    #     video = (video + 1) / 2
    #     video = (video * 255).type(torch.uint8).cpu()
    #     imageio.v3.imwrite(output_dir / f'{count}.mp4', video, fps=16)
    #     with open(output_dir / f'{count}.txt', 'w') as f:
    #         f.write(caption)
    #     if count >= 10:
    #         break
    #     count += 1
    # quit()

    if args.dump_dataset:
        # only works for flux
        import torchvision
        dataset_manager.cache(unload_models=False)
        if is_main_process():
            with torch.no_grad():
                os.makedirs(args.dump_dataset, exist_ok=True)
                vae = model.vae.to('cuda')
                train_data.post_init(
                    0,
                    1,
                    1,
                    1,
                    1,
                )
                for i, item in enumerate(train_data):
                    latents = item['latents']
                    latents = latents / vae.config.scaling_factor
                    if hasattr(vae.config, 'shift_factor') and vae.config.shift_factor is not None:
                        latents = latents + vae.config.shift_factor
                    img = vae.decode(latents.to(vae.device, vae.dtype)).sample.to(torch.float32)
                    img = img.squeeze(0)
                    img = ((img + 1) / 2).clamp(0, 1)
                    pil_img = torchvision.transforms.functional.to_pil_image(img)
                    pil_img.save(args.dump_dataset / f'{i}.png')
                    if i >= 100:
                        break
        dist.barrier()
        quit()

    dataset_manager.cache()
    if args.cache_only:
        quit()

    model.load_diffusion_model()

    # Check for top-level init_from_existing (resume workflow)
    top_level_init = config.get('init_from_existing', None)
    loaded_checkpoint_info = None

    if adapter_config := config.get('adapter', None):
        model.configure_adapter(adapter_config)
        is_adapter = True
        # Prefer adapter-level init_from_existing, fall back to top-level
        init_path = adapter_config.get('init_from_existing', top_level_init)
        if init_path:
            if is_main_process():
                print(f"[RESUME] Loading adapter weights from: {init_path}")
            loaded_checkpoint_info = model.load_adapter_weights(init_path)
    else:
        is_adapter = False
        # NOTE: We'll freeze base model parameters AFTER creating the pipeline module
        # to ensure proper gradient flow for FPS parameters

    # Store init_path for later use (FPS param loading after pipeline creation)
    init_from_existing = top_level_init or (adapter_config.get('init_from_existing', None) if adapter_config else None)

    # if this is a new run, create a new dir for it
    if not resume_from_checkpoint and is_main_process():
        run_dir = os.path.join(config['output_dir'], datetime.now(timezone.utc).strftime('%Y%m%d_%H-%M-%S'))
        os.makedirs(run_dir, exist_ok=True)
        shutil.copy(args.config, run_dir)
        shutil.copy(config['dataset'], run_dir)
        for eval_dataset in config['eval_datasets']:
            shutil.copy(eval_dataset['config'], run_dir)
    # wait for all processes then get the most recent dir (may have just been created)
    dist.barrier()
    if resume_from_checkpoint is True:  # No specific folder provided, use most recent
        run_dir = get_most_recent_run_dir(config['output_dir'])
    elif isinstance(resume_from_checkpoint, str):  # Specific folder provided
        run_dir = os.path.join(config['output_dir'], resume_from_checkpoint)
        if not os.path.exists(run_dir):
            raise ValueError(f"Checkpoint directory {run_dir} does not exist")
    else:  # Not resuming, use most recent (newly created) dir
        run_dir = get_most_recent_run_dir(config['output_dir'])

    # WandB logging
    wandb_enable = config.get('monitoring', {}).get('enable_wandb', False)
    if wandb_enable and is_main_process():
        wandb_api_key     = config['monitoring']['wandb_api_key']
        wandb_tracker     = config['monitoring']['wandb_tracker_name']
        wandb_run_name    = config['monitoring']['wandb_run_name']
        logging_dir       = run_dir
        wandb.login(key=wandb_api_key)
        wandb.init(
            project=wandb_tracker,
            name=wandb_run_name,
            config=config,
            dir=logging_dir
        )

    # Block swapping
    if blocks_to_swap := config.get('blocks_to_swap', 0):
        assert config['pipeline_stages'] == 1, 'Block swapping only works with pipeline_stages=1'
        assert 'adapter' in config, 'Block swapping only works when training LoRA'
        # Don't automatically move to GPU, we'll do that ourselves.
        def to(self, *args, **kwargs):
            pass
        deepspeed.pipe.PipelineModule.to = to
        model.enable_block_swap(blocks_to_swap)

    layers = model.to_layers()
    additional_pipeline_module_kwargs = {}
    activation_checkpointing = config['activation_checkpointing']
    if activation_checkpointing:
        if activation_checkpointing == True:
            # TODO: block swapping doesn't work with Deepspeed non-reentrant checkpoint, but PyTorch native one is fine. Some
            # weights end up on CPU where they shouldn't. Why? Are we giving anything up by not using the Deepspeed implementation?
            #checkpoint_func = deepspeed.checkpointing.non_reentrant_checkpoint
            from functools import partial
            checkpoint_func = partial(torch.utils.checkpoint.checkpoint, use_reentrant=config['reentrant_activation_checkpointing'])
        elif activation_checkpointing == 'unsloth':
            checkpoint_func = unsloth_checkpoint
        else:
            raise NotImplementedError(f'activation_checkpointing={activation_checkpointing} is not implemented')
        additional_pipeline_module_kwargs.update({
            'activation_checkpoint_interval': 1,
            'checkpointable_layers': model.checkpointable_layers,
            'activation_checkpoint_func': checkpoint_func,
        })

    num_stages = config.get('pipeline_stages', 1)
    partition_method=config.get('partition_method', 'parameters')
    partition_split = config.get('partition_split',[len(layers) / num_stages])
    pipeline_model = ManualPipelineModule(
        layers=layers,
        num_stages=num_stages,
        partition_method=partition_method,
        manual_partition_split=partition_split,
        loss_fn=model.get_loss_fn(),
        **additional_pipeline_module_kwargs
    )
    
    # MEMORY FIX: For FPS-only training, freeze base model parameters AFTER pipeline creation
    # This ensures proper gradient flow setup
    if not is_adapter:  # FPS-only training mode
        if is_main_process():
            print("[FPS_ONLY_TRAINING] Freezing base model parameters to save memory...")
        base_params_frozen = 0
        fps_params_kept = 0
        for name, param in pipeline_model.named_parameters():
            # Only freeze base model parameters, keep FPS parameters trainable
            if 'fps_conditioning' not in name and 'fps_adapter' not in name:
                param.requires_grad_(False)
                base_params_frozen += 1
            else:
                param.requires_grad_(True)  # Explicitly ensure FPS params are trainable
                fps_params_kept += 1
        if is_main_process():
            print(f"[FPS_ONLY_TRAINING] Frozen {base_params_frozen} base model parameters")
            print(f"[FPS_ONLY_TRAINING] Kept {fps_params_kept} FPS parameters trainable")
    
    # CRITICAL FIX: Enable FPS parameter gradients BEFORE optimizer creation
    # This ensures FPS parameters are included in optimizer's parameter groups
    fps_mlp_params_fixed = 0
    fps_adapter_params_fixed = 0
    fps_params_already_enabled = 0
    fps_params_disabled = 0
    
    # Debug: Check requires_grad status after freezing
    print("\n[FPS_DEBUG] Checking requires_grad status after freezing:")
    for name, param in pipeline_model.named_parameters():
        if 'fps_conditioning' in name or 'fps_adapter' in name:
            if param.requires_grad:
                fps_params_already_enabled += 1
            else:
                fps_params_disabled += 1
                
    print(f"  FPS params with requires_grad=True: {fps_params_already_enabled}")
    print(f"  FPS params with requires_grad=False: {fps_params_disabled}")
    
    # Double-check: Make absolutely sure FPS parameters have gradients enabled
    for name, param in pipeline_model.named_parameters():
        if 'fps_conditioning' in name:
            if not param.requires_grad:
                param.requires_grad_(True)
                fps_mlp_params_fixed += 1
        elif 'fps_adapter' in name:
            if not param.requires_grad:
                param.requires_grad_(True)
                fps_adapter_params_fixed += 1
    
    if fps_mlp_params_fixed > 0 or fps_adapter_params_fixed > 0:
        print(f'[FPS_FIX] Had to re-enable gradients for {fps_mlp_params_fixed} FPS MLP + {fps_adapter_params_fixed} FPS adapter parameters')

    # CRITICAL FIX: Reload FPS adapters into pipeline model after pipeline creation
    # The pipeline model has NEW FPS adapter instances that need to be loaded from checkpoint
    if loaded_checkpoint_info and init_from_existing and loaded_checkpoint_info.get('has_fps_adapter'):
        if is_main_process():
            print(f"\n[RESUME_FIX] Reloading FPS adapters into pipeline model...")
            print(f"[RESUME_FIX] Loading from: {init_from_existing}")

        # Load FPS adapters from checkpoint into pipeline model
        import safetensors
        from pathlib import Path

        checkpoint_dir = Path(init_from_existing)
        if checkpoint_dir.is_dir():
            safetensors_files = list(checkpoint_dir.glob('*.safetensors'))
            if safetensors_files:
                checkpoint_file = safetensors_files[0]
                checkpoint = safetensors.torch.load_file(str(checkpoint_file))

                # Build a mapping from checkpoint keys to values
                # Strip 'diffusion_model.' prefix if present
                fps_checkpoint_params = {}
                for k, v in checkpoint.items():
                    if 'fps_conditioning' in k or 'fps_adapter' in k:
                        if k.startswith('diffusion_model.'):
                            k = k.replace('diffusion_model.', '', 1)
                        fps_checkpoint_params[k] = v

                # Match parameters by name suffix (e.g., 'blocks.27.fps_adapter.k_fps_up.weight')
                # and load them into the pipeline model
                params_loaded = 0
                params_mismatched = 0

                for pipeline_name, pipeline_param in pipeline_model.named_parameters():
                    # Check if this is an FPS parameter
                    if 'fps_conditioning' in pipeline_name or 'fps_adapter' in pipeline_name:
                        # Try to find matching checkpoint parameter
                        # The pipeline name might be like '_module_1.block.fps_adapter.k_fps_up.weight'
                        # and checkpoint name might be 'blocks.27.fps_adapter.k_fps_up.weight'

                        # Extract the fps_adapter/fps_conditioning part onwards
                        if 'fps_adapter' in pipeline_name:
                            suffix_start = pipeline_name.find('fps_adapter')
                        elif 'fps_conditioning' in pipeline_name:
                            suffix_start = pipeline_name.find('fps_conditioning')
                        else:
                            continue

                        suffix = pipeline_name[suffix_start:]

                        # Find checkpoint param with matching suffix
                        matched_value = None
                        for ckpt_name, ckpt_value in fps_checkpoint_params.items():
                            if ckpt_name.endswith(suffix):
                                matched_value = ckpt_value
                                break

                        if matched_value is not None:
                            pipeline_param.data.copy_(matched_value)
                            params_loaded += 1
                        else:
                            params_mismatched += 1

                if is_main_process():
                    print(f"[RESUME_FIX] Loaded {params_loaded} FPS parameters into pipeline model")
                    if params_mismatched > 0:
                        print(f"[RESUME_FIX] WARNING: {params_mismatched} FPS parameters not found in checkpoint")

    # RESUME WORKFLOW: Freeze loaded parameters based on explicit config options
    # This runs for both top-level and adapter-level init_from_existing
    # Backward compatible: defaults (all freeze=False) match old behavior + add reinitialization
    if loaded_checkpoint_info and init_from_existing:
        # Get explicit freeze options from config
        freeze_base_lora_option = False
        freeze_fps_mlp_option = False
        freeze_fps_adapters_option = False

        if adapter_config:
            freeze_base_lora_option = adapter_config.get('freeze_base_lora', False)

        model_config = config.get('model', {})
        freeze_fps_mlp_option = model_config.get('freeze_fps_mlp', False)
        freeze_fps_adapters_option = model_config.get('freeze_fps_adapters', False)

        if is_main_process():
            print(f"\n{'='*80}")
            print(f"[RESUME] Checkpoint Resume Workflow")
            print(f"{'='*80}")
            print(f"[RESUME] Checkpoint path: {init_from_existing}")
            print(f"\n[RESUME] Checkpoint contents:")
            print(f"  - Base LoRA: {'✓ LOADED' if loaded_checkpoint_info['has_base_lora'] else '✗ Not present'}")
            print(f"  - FPS MLP: {'✓ LOADED' if loaded_checkpoint_info['has_fps_mlp'] else '✗ Not present'}")
            print(f"  - FPS Adapters: {'✓ LOADED' if loaded_checkpoint_info['has_fps_adapter'] else '✗ Not present'}")
            print(f"\n[RESUME] Freeze configuration:")
            print(f"  - freeze_base_lora: {freeze_base_lora_option}")
            print(f"  - freeze_fps_mlp: {freeze_fps_mlp_option}")
            print(f"  - freeze_fps_adapters: {freeze_fps_adapters_option}")
            print(f"\n[RESUME] Applying selective parameter freezing...")

        # Apply freezing only if explicitly requested AND params exist in checkpoint
        freeze_base_lora = loaded_checkpoint_info['has_base_lora'] and freeze_base_lora_option
        freeze_fps_mlp = loaded_checkpoint_info['has_fps_mlp'] and freeze_fps_mlp_option
        freeze_fps_adapters = loaded_checkpoint_info['has_fps_adapter'] and freeze_fps_adapters_option

        params_frozen = 0
        base_lora_frozen = 0
        fps_mlp_frozen = 0
        fps_adapters_frozen = 0

        for name, param in pipeline_model.named_parameters():
            should_freeze = False

            # Freeze base LoRA if loaded and freeze option enabled
            if freeze_base_lora and '.lora_' in name and 'fps' not in name:
                should_freeze = True
                base_lora_frozen += 1

            # Freeze FPS MLP if loaded and freeze option enabled
            if freeze_fps_mlp and 'fps_conditioning' in name:
                should_freeze = True
                fps_mlp_frozen += 1

            # Freeze FPS adapters if loaded and freeze option enabled
            if freeze_fps_adapters and 'fps_adapter' in name:
                should_freeze = True
                fps_adapters_frozen += 1

            if should_freeze and param.requires_grad:
                param.requires_grad_(False)
                params_frozen += 1

        if is_main_process():
            if params_frozen > 0:
                print(f"\n[RESUME] Freezing summary:")
                print(f"  - Total parameters frozen: {params_frozen}")
                if base_lora_frozen > 0:
                    print(f"  - Base LoRA: {base_lora_frozen} params frozen")
                if fps_mlp_frozen > 0:
                    print(f"  - FPS MLP: {fps_mlp_frozen} params frozen")
                if fps_adapters_frozen > 0:
                    print(f"  - FPS Adapters: {fps_adapters_frozen} params frozen")
            else:
                print(f"\n[RESUME] No parameters frozen (all will be trained)")

        # Reinitialize NEW FPS parameters that weren't in checkpoint
        # This ensures proper zero-disturbance initialization for new FPS adapters
        if not loaded_checkpoint_info['has_fps_adapter'] and not freeze_fps_adapters_option:
            # NEW FPS adapters were created but not loaded from checkpoint - reinitialize them
            fps_adapters_reinitialized = 0
            if is_main_process():
                print(f"\n[RESUME] Reinitializing NEW FPS adapters (not in checkpoint)...")

            for name, module in pipeline_model.named_modules():
                if 'fps_adapter' in name and hasattr(module, '_init_lora_weights'):
                    module._init_lora_weights()
                    fps_adapters_reinitialized += 1

            if is_main_process() and fps_adapters_reinitialized > 0:
                print(f"[RESUME] ✓ Reinitialized {fps_adapters_reinitialized} NEW FPS adapters")

        # Reinitialize FPS MLP if not in checkpoint
        if not loaded_checkpoint_info['has_fps_mlp'] and not freeze_fps_mlp_option:
            fps_mlp_reinitialized = False
            if is_main_process():
                print(f"\n[RESUME] Reinitializing NEW FPS MLP (not in checkpoint)...")

            for name, module in pipeline_model.named_modules():
                if 'fps_conditioning' in name and hasattr(module, '_init_weights'):
                    module._init_weights()
                    fps_mlp_reinitialized = True
                    break

            if is_main_process() and fps_mlp_reinitialized:
                print(f"[RESUME] ✓ Reinitialized FPS MLP")

        # Final summary: Show what will be trained
        if is_main_process():
            trainable_base_lora = 0
            trainable_fps_mlp = 0
            trainable_fps_adapters = 0

            for name, param in pipeline_model.named_parameters():
                if param.requires_grad:
                    if 'fps_conditioning' in name:
                        trainable_fps_mlp += 1
                    elif 'fps_adapter' in name:
                        trainable_fps_adapters += 1
                    elif '.lora_' in name or '.default.' in name:
                        trainable_base_lora += 1

            print(f"\n[RESUME] Training summary:")
            print(f"  - Base LoRA: {trainable_base_lora} params {'(TRAINING)' if trainable_base_lora > 0 else '(FROZEN)'}")
            print(f"  - FPS MLP: {trainable_fps_mlp} params {'(TRAINING)' if trainable_fps_mlp > 0 else '(FROZEN)'}")
            print(f"  - FPS Adapters: {trainable_fps_adapters} params {'(TRAINING)' if trainable_fps_adapters > 0 else '(FROZEN)'}")

            # Show initial parameter magnitudes
            print(f"\n[RESUME] Initial parameter magnitudes:")

            # Base LoRA sample
            for name, param in pipeline_model.named_parameters():
                if '.lora_' in name and 'fps' not in name and 'default' in name:
                    norm = param.data.norm().item()
                    mean = param.data.mean().item()
                    print(f"  Base LoRA (sample {name[:60]}...): norm={norm:.6f}, mean={mean:.6f}, grad={param.requires_grad}")
                    break

            # FPS MLP
            for name, param in pipeline_model.named_parameters():
                if 'fps_conditioning.lin2.weight' in name:
                    norm = param.data.norm().item()
                    mean = param.data.mean().item()
                    print(f"  FPS MLP lin2: norm={norm:.6f}, mean={mean:.6f}, grad={param.requires_grad}")
                    break

            # FPS adapters sample
            for name, param in pipeline_model.named_parameters():
                if 'fps_adapter' in name and 'k_fps_up.weight' in name:
                    norm = param.data.norm().item()
                    mean = param.data.mean().item()
                    print(f"  FPS Adapter (sample {name[:60]}...): norm={norm:.6f}, mean={mean:.6f}, grad={param.requires_grad}")
                    break

            print(f"{'='*80}\n")

    # Debug: Final verification
    fps_params_enabled_after = 0
    for name, param in pipeline_model.named_parameters():
        if 'fps_conditioning' in name or 'fps_adapter' in name:
            if param.requires_grad:
                fps_params_enabled_after += 1
                
    print(f"[FPS_DEBUG] Final check: {fps_params_enabled_after} FPS params have requires_grad=True")
    
    parameters_to_train = [p for p in pipeline_model.parameters() if p.requires_grad]
    
    # Debug: Count parameters to train
    print(f"\n[FPS_DEBUG] Total parameters with requires_grad=True: {len(parameters_to_train)}")
    fps_params_in_training = 0
    for p in parameters_to_train:
        # Check if this parameter is an FPS parameter by checking its shape/size
        # FPS MLP parameters have specific shapes we can check
        if len(p.shape) == 2:  # 2D tensor (weight matrices)
            if p.shape == (64, 1):  # lin1.weight shape for FPS MLP
                fps_params_in_training += 1
            elif p.shape == (3072, 64):  # lin2.weight shape for FPS MLP
                fps_params_in_training += 1
            elif p.shape[0] == 32 or p.shape[1] == 32:  # LoRA matrices with rank 32
                fps_params_in_training += 1
        elif len(p.shape) == 1:  # 1D tensor (biases)
            if p.shape[0] == 64 or p.shape[0] == 3072:  # FPS MLP biases
                fps_params_in_training += 1
        elif p.numel() == 1:  # gate_alpha is a single value
            fps_params_in_training += 1
            
    print(f"[FPS_DEBUG] Estimated FPS parameters in training list: {fps_params_in_training}")

    if config['compile']:
        pipeline_model.compile()

    def get_optimizer(model_parameters):
        if len(model_parameters) == 0:
            return DummyOptimizer()

        optim_config = config['optimizer']
        optim_type = optim_config['type']
        optim_type_lower = optim_type.lower()

        args = []
        kwargs = {k: v for k, v in optim_config.items() if k not in ['type', 'gradient_release']}

        if optim_type_lower == 'adamw':
            # TODO: fix this. I'm getting "fatal error: cuda_runtime.h: No such file or directory"
            # when Deepspeed tries to build the fused Adam extension.
            # klass = deepspeed.ops.adam.FusedAdam
            klass = torch.optim.AdamW
        elif optim_type_lower == 'adamw8bit':
            import bitsandbytes
            klass = bitsandbytes.optim.AdamW8bit
        elif optim_type_lower == 'adamw_optimi':
            import optimi
            klass = optimi.AdamW
        elif optim_type_lower == 'stableadamw':
            import optimi
            klass = optimi.StableAdamW
        elif optim_type_lower == 'sgd':
            klass = torch.optim.SGD
        elif optim_type_lower == 'adamw8bitkahan':
            from optimizers import adamw_8bit
            klass = adamw_8bit.AdamW8bitKahan
        elif optim_type_lower == 'offload':
            from torchao.prototype.low_bit_optim import CPUOffloadOptimizer
            klass = CPUOffloadOptimizer
            args.append(torch.optim.AdamW)
            kwargs['fused'] = True
        elif optim_type_lower == 'automagic':
            from optimizers import automagic
            klass = automagic.Automagic
        elif optim_type_lower == 'genericoptim':
            from optimizers import generic_optim
            klass = generic_optim.GenericOptim
        else:
            import pytorch_optimizer
            klass = getattr(pytorch_optimizer, optim_type)

        if optim_config.get('gradient_release', False):
            # Prevent deepspeed from logging every single param group lr
            def _report_progress(self, step):
                lr = self.get_lr()
                mom = self.get_mom()
                deepspeed.utils.logging.log_dist(f"step={step}, skipped={self.skipped_steps}, lr={lr[0]}, mom={mom[0]}", ranks=[0])
            deepspeed.runtime.engine.DeepSpeedEngine._report_progress = _report_progress

            # Deepspeed executes all the code to reduce grads across data parallel ranks even if the DP world size is 1.
            # As part of this, any grads that are None are set to zeros. We're doing gradient release to save memory,
            # so we have to avoid this.
            def _exec_reduce_grads(self):
                assert self.mpu.get_data_parallel_world_size() == 1, 'When using gradient release, data parallel world size must be 1. Make sure pipeline_stages = num_gpus.'
                return
            deepspeed.runtime.pipe.engine.PipelineEngine._INSTRUCTION_MAP[deepspeed.runtime.pipe.schedule.ReduceGrads] = _exec_reduce_grads

            # When pipelining multiple forward and backward passes, normally updating the parameter in-place causes an error when calling
            # backward() on future micro-batches. But we can modify .data directly so the autograd engine doesn't detect in-place modifications.
            # TODO: this is unbelievably hacky and not mathematically sound, I'm just seeing if it works at all.
            def add_(self, *args, **kwargs):
                self.data.add_(*args, **kwargs)
            for p in model_parameters:
                p.add_ = add_.__get__(p)

            if 'foreach' in inspect.signature(klass).parameters:
                kwargs['foreach'] = False

            # We're doing an optimizer step for each micro-batch. Scale momentum and EMA betas so that the contribution
            # decays at the same rate it would if we were doing one step per batch like normal.
            # Reference: https://alexeytochin.github.io/posts/batch_size_vs_momentum/batch_size_vs_momentum.html
            gas = ds_config['gradient_accumulation_steps']
            if 'betas' in kwargs:
                for i in range(len(kwargs['betas'])):
                    kwargs['betas'][i] = kwargs['betas'][i] ** (1/gas)
            if 'momentum' in kwargs:
                kwargs['momentum'] = kwargs['momentum'] ** (1/gas)

            optimizer_dict = {}
            for pg in model.get_param_groups(model_parameters):
                param_kwargs = kwargs.copy()
                if isinstance(pg, dict):
                    # param group
                    for p in pg['params']:
                        param_kwargs['lr'] = pg['lr']
                        optimizer_dict[p] = klass([p], **param_kwargs)
                else:
                    # param
                    optimizer_dict[pg] = klass([pg], **param_kwargs)

            def optimizer_hook(p):
                optimizer_dict[p].step()
                optimizer_dict[p].zero_grad()

            for p in model_parameters:
                p.register_post_accumulate_grad_hook(optimizer_hook)

            from optimizers import gradient_release
            return gradient_release.GradientReleaseOptimizerWrapper(list(optimizer_dict.values()))
        elif optim_type_lower == 'genericoptim':
            kwargs['compile'] = config['compile']
            new_param_groups = []
            param_groups = model.get_param_groups(model_parameters)
            for pg in param_groups:
                params = pg.pop('params')
                params_2d = []
                params_other = []
                for p in params:
                    if p.ndim == 2:
                        params_2d.append(p)
                    else:
                        params_other.append(p)
                pg_2d = pg.copy()
                pg_2d['params'] = params_2d
                if kwargs.get('second_moment_type', None) == 'sn':
                    pg_2d['subset_size'] = 'heuristics'
                for key in ('rank', 'proj_type', 'update_proj_gap'):
                    if key in kwargs:
                        pg_2d[key] = kwargs.pop(key)
                new_param_groups.append(pg_2d)
                pg_other = pg
                pg_other['params'] = params_other
                new_param_groups.append(pg_other)
            return klass(new_param_groups, *args, **kwargs)
        else:
            param_groups = model.get_param_groups(model_parameters)
            return klass(param_groups, *args, **kwargs)

    # Debug: Check what parameters are being passed to optimizer
    print(f"\n[OPTIMIZER_DEBUG] Creating optimizer with {len(parameters_to_train)} parameters")
    fps_params_count = 0
    for p in parameters_to_train:
        # Try to identify FPS parameters by their shapes
        if len(p.shape) == 2:  # 2D tensors
            if p.shape == (64, 1) or p.shape == (3072, 64):  # FPS MLP weights
                fps_params_count += 1
            elif p.shape[0] == 32 or p.shape[1] == 32:  # LoRA matrices with rank 32
                fps_params_count += 1
        elif len(p.shape) == 1:  # 1D tensors
            if p.shape[0] == 64 or p.shape[0] == 3072:  # FPS MLP biases
                fps_params_count += 1
        elif p.numel() == 1:  # gate_alpha
            fps_params_count += 1
    print(f"[OPTIMIZER_DEBUG] Estimated {fps_params_count} FPS parameters in optimizer")
    
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=pipeline_model,
        model_parameters=parameters_to_train,
        optimizer=get_optimizer,
        config=ds_config,
    )
    model.model_engine = model_engine
    
    # Debug: Check optimizer parameter groups after initialization
    print("\n[OPTIMIZER_DEBUG] After DeepSpeed initialization:")
    if hasattr(optimizer, 'param_groups'):
        total_params_in_optimizer = 0
        for i, group in enumerate(optimizer.param_groups):
            num_params = len(group['params'])
            total_params_in_optimizer += num_params
            print(f"  Param group {i}: {num_params} parameters, lr={group['lr']}")
        print(f"  Total parameters in optimizer: {total_params_in_optimizer}")
    else:
        print("  Optimizer doesn't have param_groups attribute")
    
    # Debug: Check if FPS parameters are in the pipeline module after DeepSpeed init
    fps_params_after_init = 0
    fps_params_requires_grad = 0
    for name, param in model_engine.module.named_parameters():
        if 'fps_conditioning' in name or 'fps_adapter' in name:
            fps_params_after_init += 1
            if param.requires_grad:
                fps_params_requires_grad += 1
    print(f"\n[FPS_DEBUG] After DeepSpeed init:")
    print(f"  Found {fps_params_after_init} FPS parameters in pipeline module")
    print(f"  {fps_params_requires_grad} have requires_grad=True")
    
    # --- Re-apply FPS initialization for pipeline parallel ---
    # In pipeline parallel mode, we need to re-initialize through the pipeline module
    # BUT: Skip this if we loaded FPS parameters from checkpoint!
    skip_fps_reinit = loaded_checkpoint_info and init_from_existing and (
        loaded_checkpoint_info.get('has_fps_adapter') or loaded_checkpoint_info.get('has_fps_mlp')
    )

    if not skip_fps_reinit:
        print("\n[FPS_INIT] Re-initializing FPS parameters after DeepSpeed...")
        fps_mlp_reinitialized = 0
        fps_adapter_reinitialized = 0

        for name, module in model_engine.module.named_modules():
            if 'fps_conditioning' in name and hasattr(module, '_init_weights'):
                module._init_weights()
                fps_mlp_reinitialized += 1
            elif 'fps_adapter' in name and hasattr(module, '_init_lora_weights'):
                module._init_lora_weights()
                fps_adapter_reinitialized += 1

        print(f"[FPS_INIT] Re-initialized {fps_mlp_reinitialized} FPS MLP modules")
        print(f"[FPS_INIT] Re-initialized {fps_adapter_reinitialized} FPS adapter modules")
    else:
        print("\n[FPS_INIT] Skipping FPS reinitialization (loaded from checkpoint)")

        # CRITICAL FIX: Load FPS parameters into DeepSpeed model AFTER initialization
        # The model state was copied during deepspeed.initialize(), so we need to load again
        if is_main_process():
            print(f"\n[RESUME_FIX_V2] Reloading FPS adapters into DeepSpeed model...")
            print(f"[RESUME_FIX_V2] Loading from: {init_from_existing}")

        from pathlib import Path
        import safetensors
        checkpoint_dir = Path(init_from_existing)
        if checkpoint_dir.is_dir():
            safetensors_files = list(checkpoint_dir.glob('*.safetensors'))
            if safetensors_files:
                checkpoint_file = safetensors_files[0]
                checkpoint = safetensors.torch.load_file(str(checkpoint_file))

                # Build a mapping from checkpoint keys to values
                fps_checkpoint_params = {}
                for k, v in checkpoint.items():
                    if 'fps_conditioning' in k or 'fps_adapter' in k:
                        if k.startswith('diffusion_model.'):
                            k = k.replace('diffusion_model.', '', 1)
                        fps_checkpoint_params[k] = v

                # Load into DeepSpeed model engine's module
                params_loaded = 0
                params_mismatched = 0

                for pipeline_name, pipeline_param in model_engine.module.named_parameters():
                    if 'fps_conditioning' in pipeline_name or 'fps_adapter' in pipeline_name:
                        # Extract block number and parameter name from pipeline name
                        # Pipeline name might be like: '_module_1.block.fps_adapter.k_fps_down.weight'
                        # Checkpoint name is like: 'blocks.27.fps_adapter.k_fps_down.weight'

                        # Try to extract block number from pipeline name
                        import re
                        # Look for patterns like 'block' followed by index in module path
                        # The actual block number is encoded in the layer structure

                        # For now, match by extracting the FPS param name and trying to find
                        # corresponding checkpoint params by trying different block numbers
                        if 'fps_adapter' in pipeline_name:
                            param_suffix = pipeline_name[pipeline_name.find('fps_adapter'):]
                        elif 'fps_conditioning' in pipeline_name:
                            param_suffix = pipeline_name[pipeline_name.find('fps_conditioning'):]
                        else:
                            continue

                        # Try to find the exact match by looking at all checkpoint params
                        # that end with this suffix, and match based on order
                        matched_value = None

                        # Get all checkpoint params with this suffix
                        matching_ckpt_params = []
                        for ckpt_name, ckpt_value in fps_checkpoint_params.items():
                            if ckpt_name.endswith(param_suffix):
                                # Extract block number from checkpoint name
                                block_match = re.search(r'blocks\.(\d+)\.', ckpt_name)
                                block_num = int(block_match.group(1)) if block_match else -1
                                matching_ckpt_params.append((block_num, ckpt_name, ckpt_value))

                        if matching_ckpt_params:
                            # Sort by block number
                            matching_ckpt_params.sort(key=lambda x: x[0])

                            # Try to determine which block this pipeline param belongs to
                            # by looking at its position in the module hierarchy
                            # For now, let's try to extract layer/module index from pipeline name
                            layer_match = re.search(r'_module_(\d+)|layer.?(\d+)|\.(\d+)\.', pipeline_name)

                            # If we can't determine the exact block, we need a different strategy
                            # Let's count FPS adapter params seen so far to determine index
                            if not hasattr(model_engine, '_fps_param_counter'):
                                model_engine._fps_param_counter = {}

                            # Use param suffix as key to track which occurrence this is
                            if param_suffix not in model_engine._fps_param_counter:
                                model_engine._fps_param_counter[param_suffix] = 0

                            idx = model_engine._fps_param_counter[param_suffix]
                            if idx < len(matching_ckpt_params):
                                matched_value = matching_ckpt_params[idx][2]
                                model_engine._fps_param_counter[param_suffix] += 1

                        if matched_value is not None:
                            pipeline_param.data.copy_(matched_value)
                            params_loaded += 1
                        else:
                            params_mismatched += 1

                if is_main_process():
                    print(f"[RESUME_FIX_V2] Loaded {params_loaded} FPS parameters into DeepSpeed model")
                    if params_mismatched > 0:
                        print(f"[RESUME_FIX_V2] WARNING: {params_mismatched} FPS parameters not found in checkpoint")
    
    # Verify initialization (for adapters that have verify method)
    verified_adapters = 0
    for name, module in model_engine.module.named_modules():
        if 'fps_adapter' in name and hasattr(module, 'verify_initialization'):
            if module.verify_initialization():
                verified_adapters += 1
    
    if verified_adapters > 0:
        print(f"[FPS_INIT] Verified {verified_adapters} FPS adapters have correct initialization")
    
    # Debug RMSNorm initialization magnitudes
    rmsnorm_count = 0
    for name, module in model_engine.module.named_modules():
        if 'fps_adapter' in name and hasattr(module, 'norm_k_fps'):
            norm_weight_magnitude = module.norm_k_fps.weight.abs().mean().item()
            print(f"[FPS_DEBUG] {name}.norm_k_fps.weight magnitude = {norm_weight_magnitude:.6f}")
            rmsnorm_count += 1
    
    if rmsnorm_count > 0:
        print(f"[FPS_DEBUG] Found {rmsnorm_count} RMSNorm layers in FPS adapters")

    
    if model_engine.is_pipe_parallel:
         grid = model_engine.grid
         model_engine.first_last_stage_group = dist.new_group(ranks=[grid.pp_group[0], grid.pp_group[-1]])

    lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
    if config['warmup_steps'] > 0:
        warmup_steps = config['warmup_steps']
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1/warmup_steps, total_iters=warmup_steps)
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, lr_scheduler], milestones=[warmup_steps])
    model_engine.lr_scheduler = lr_scheduler

    train_data.post_init(
        model_engine.grid.get_data_parallel_rank(),
        model_engine.grid.get_data_parallel_world_size(),
        model_engine.train_micro_batch_size_per_gpu(),
        model_engine.gradient_accumulation_steps(),
        config.get('image_micro_batch_size_per_gpu', model_engine.train_micro_batch_size_per_gpu()),
    )
    for eval_data in eval_data_map.values():
        eval_data.post_init(
            model_engine.grid.get_data_parallel_rank(),
            model_engine.grid.get_data_parallel_world_size(),
            config.get('eval_micro_batch_size_per_gpu', model_engine.train_micro_batch_size_per_gpu()),
            config['eval_gradient_accumulation_steps'],
            config.get('image_eval_micro_batch_size_per_gpu', config.get('eval_micro_batch_size_per_gpu', model_engine.train_micro_batch_size_per_gpu())),
        )

    # Might be useful because we set things in fp16 / bf16 without explicitly enabling Deepspeed fp16 mode.
    # Unsure if really needed.
    communication_data_type = config['lora']['dtype'] if 'lora' in config else config['model']['dtype']
    model_engine.communication_data_type = communication_data_type

    train_dataloader = dataset_util.PipelineDataLoader(train_data, model_engine, model_engine.gradient_accumulation_steps(), model)

    step = 1
    # make sure to do this before calling model_engine.set_dataloader(), as that method creates an iterator
    # which starts creating dataloader internal state
    if resume_from_checkpoint:
        load_path, client_state = model_engine.load_checkpoint(
            run_dir,
            load_module_strict=False,
            load_lr_scheduler_states='force_constant_lr' not in config,
        )
        dist.barrier()  # just so the print below doesn't get swamped
        assert load_path is not None
        train_dataloader.load_state_dict(client_state['custom_loader'])
        step = client_state['step'] + 1
        del client_state
        if is_main_process():
            print(f'Resuming training from checkpoint. Resuming at epoch: {train_dataloader.epoch}, step: {step}')

    if 'force_constant_lr' in config:
        model_engine.lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
        for pg in optimizer.param_groups:
            pg['lr'] = config['force_constant_lr']

    steps_per_epoch = len(train_dataloader) // model_engine.gradient_accumulation_steps()
    model_engine.total_steps = steps_per_epoch * config['epochs']

    eval_dataloaders = {
        name: dataset_util.PipelineDataLoader(eval_data, model_engine, config['eval_gradient_accumulation_steps'], model, num_dataloader_workers=0)
        for name, eval_data in eval_data_map.items()
    }

    epoch = train_dataloader.epoch
    tb_writer = SummaryWriter(log_dir=run_dir) if is_main_process() else None
    saver = utils.saver.Saver(args, config, is_adapter, run_dir, model, train_dataloader, model_engine, pipeline_model)

    disable_block_swap_for_eval = config.get('disable_block_swap_for_eval', False)
    if config['eval_before_first_step'] and not resume_from_checkpoint:
        evaluate(model, model_engine, eval_dataloaders, tb_writer, 0, config['eval_gradient_accumulation_steps'], disable_block_swap_for_eval)

    # TODO: this is state we need to save and resume when resuming from checkpoint. It only affects logging.
    # Function to log FPS parameter magnitudes for debugging
    def log_fps_param_magnitudes(model_engine, epoch):
        """Log magnitudes of FPS parameters at the end of each epoch"""
        if not is_main_process():
            return
            
        print(f"\n[FPS_MAGNITUDE] Epoch {epoch} - Checking pipeline model parameters:")

        # In pipeline parallel mode, we need to iterate through the pipeline module's parameters
        base_lora_params = {}
        fps_mlp_params = {}
        fps_adapter_params = {}

        for name, param in model_engine.module.named_parameters():
            if 'fps_conditioning' in name:
                if not param.is_meta:
                    fps_mlp_params[name] = {
                        'mean': param.data.abs().mean().item(),
                        'std': param.data.std().item(),
                        'max': param.data.abs().max().item(),
                        'requires_grad': param.requires_grad
                    }
            elif 'fps_adapter' in name:
                if not param.is_meta:
                    fps_adapter_params[name] = {
                        'mean': param.data.abs().mean().item(),
                        'std': param.data.std().item(),
                        'max': param.data.abs().max().item(),
                        'requires_grad': param.requires_grad
                    }
            elif ('.lora_' in name or '.default.' in name) and 'fps' not in name:
                # Base LoRA parameters
                if not param.is_meta and len(base_lora_params) < 3:  # Sample first 3
                    base_lora_params[name] = {
                        'mean': param.data.abs().mean().item(),
                        'std': param.data.std().item(),
                        'max': param.data.abs().max().item(),
                        'requires_grad': param.requires_grad
                    }
        
        # Print Base LoRA parameters (sample) with training status
        if base_lora_params:
            print("  Base LoRA Parameters (sample):")
            for name, stats in base_lora_params.items():
                short_name = name.split('.')[-4:]  # Show more context for LoRA
                short_name = '.'.join(short_name)
                if len(short_name) > 60:
                    short_name = short_name[-60:]
                status = "TRAINING" if stats['requires_grad'] else "FROZEN"
                print(f"    {short_name}: mean={stats['mean']:.6f}, std={stats['std']:.6f}, max={stats['max']:.6f} [{status}]")

        # Print FPS MLP parameters with training status
        if fps_mlp_params:
            print("  FPS MLP Parameters:")
            for name, stats in fps_mlp_params.items():
                short_name = name.split('.')[-2:]
                short_name = '.'.join(short_name)
                status = "TRAINING" if stats['requires_grad'] else "FROZEN"
                print(f"    {short_name}: mean={stats['mean']:.6f}, std={stats['std']:.6f}, max={stats['max']:.6f} [{status}]")

        # Print FPS adapter parameters (sample a few) with training status
        if fps_adapter_params:
            print("  FPS Adapter Parameters (sample):")
            count = 0
            for name, stats in fps_adapter_params.items():
                if count >= 10:  # Only show first 10 for brevity
                    print(f"    ... and {len(fps_adapter_params) - 10} more")
                    break
                short_name = name.split('.')[-3:]
                short_name = '.'.join(short_name)
                status = "TRAINING" if stats['requires_grad'] else "FROZEN"
                print(f"    {short_name}: mean={stats['mean']:.6f}, std={stats['std']:.6f}, max={stats['max']:.6f} [{status}]")
                count += 1
        
        if not fps_mlp_params and not fps_adapter_params:
            print("  WARNING: No FPS parameters found in pipeline model!")
            return
            
        print(f"\n[FPS_MAGNITUDE] Epoch {epoch} Parameter Magnitudes:")
        
        # Track FPS conditioning MLP parameters
        if hasattr(model.transformer, 'fps_conditioning'):
            fps_cond = model.transformer.fps_conditioning
            if hasattr(fps_cond, 'lin1') and not fps_cond.lin1.weight.is_meta:
                w1_mag = fps_cond.lin1.weight.data.abs().mean().item()
                b1_mag = fps_cond.lin1.bias.data.abs().mean().item() if fps_cond.lin1.bias is not None else 0
                print(f"  FPS MLP lin1: weight={w1_mag:.6f}, bias={b1_mag:.6f}")
            if hasattr(fps_cond, 'lin2') and not fps_cond.lin2.weight.is_meta:
                w2_mag = fps_cond.lin2.weight.data.abs().mean().item()
                b2_mag = fps_cond.lin2.bias.data.abs().mean().item() if fps_cond.lin2.bias is not None else 0
                print(f"  FPS MLP lin2: weight={w2_mag:.6f}, bias={b2_mag:.6f}")
        
        # Track FPS adapter parameters (sample a few blocks)
        adapter_stats = {'gate_alpha': [], 'k_down': [], 'v_down': [], 'k_up': [], 'v_up': []}
        blocks_to_sample = [27, 30, 35, 39]  # Sample a few blocks across the range
        
        for block_idx in blocks_to_sample:
            if block_idx < len(model.transformer.blocks):
                block = model.transformer.blocks[block_idx]
                if hasattr(block, 'fps_adapter') and block.fps_adapter is not None:
                    adapter = block.fps_adapter
                    # Handle different gate modes
                    if adapter.gate_mode == 'learned' and hasattr(adapter, 'gate_alpha') and not adapter.gate_alpha.is_meta:
                        # Learned gate: gate_alpha is trainable parameter
                        gate_val = adapter.gate_alpha.item()
                        gate_sigmoid = torch.sigmoid(adapter.gate_alpha).item()
                        adapter_stats['gate_alpha'].append((block_idx, gate_val, gate_sigmoid))
                    elif adapter.gate_mode == 'fixed' and hasattr(adapter, 'gate_fixed'):
                        # Fixed gate: gate_fixed is constant buffer
                        gate_val = adapter.gate_fixed.item()
                        gate_sigmoid = gate_val  # No sigmoid needed for fixed mode
                        adapter_stats['gate_alpha'].append((block_idx, f"fixed:{gate_val}", gate_sigmoid))
                    
                    # LoRA weights magnitudes
                    if not adapter.k_fps_down.weight.is_meta:
                        k_down_mag = adapter.k_fps_down.weight.data.abs().mean().item()
                        v_down_mag = adapter.v_fps_down.weight.data.abs().mean().item()
                        k_up_mag = adapter.k_fps_up.weight.data.abs().mean().item()
                        v_up_mag = adapter.v_fps_up.weight.data.abs().mean().item()
                        
                        adapter_stats['k_down'].append((block_idx, k_down_mag))
                        adapter_stats['v_down'].append((block_idx, v_down_mag))
                        adapter_stats['k_up'].append((block_idx, k_up_mag))
                        adapter_stats['v_up'].append((block_idx, v_up_mag))
        
        # Print adapter statistics
        if adapter_stats['gate_alpha']:
            print(f"  FPS Adapter Gates (alpha -> sigmoid):")
            for block_idx, alpha, sigmoid in adapter_stats['gate_alpha']:
                # Handle both numeric and string alpha values (for fixed vs learned gates)
                if isinstance(alpha, str):
                    print(f"    Block {block_idx}: {alpha} -> gate={sigmoid:.4f}")
                else:
                    print(f"    Block {block_idx}: alpha={alpha:.4f} -> gate={sigmoid:.4f}")
        
        if adapter_stats['k_down']:
            print(f"  FPS Adapter LoRA Down (A) magnitudes:")
            for i, (block_idx, k_mag) in enumerate(adapter_stats['k_down']):
                v_mag = adapter_stats['v_down'][i][1]
                print(f"    Block {block_idx}: k_down={k_mag:.6f}, v_down={v_mag:.6f}")
            
            print(f"  FPS Adapter LoRA Up (B) magnitudes:")
            for i, (block_idx, k_mag) in enumerate(adapter_stats['k_up']):
                v_mag = adapter_stats['v_up'][i][1]
                print(f"    Block {block_idx}: k_up={k_mag:.6f}, v_up={v_mag:.6f}")
    
    def debug_fps_param_updates(model_engine, step):
        """Debug function to track FPS parameter updates and gradients during training"""
        if not is_main_process():
            return
            
        # For pipeline parallel, check parameters directly from the pipeline module
        fps_params_with_grad = 0
        fps_params_without_grad = 0
        fps_param_samples = []
        
        for name, param in model_engine.module.named_parameters():
            if 'fps_conditioning' in name or 'fps_adapter' in name:
                if param.grad is not None:
                    fps_params_with_grad += 1
                    if len(fps_param_samples) < 5:  # Sample first 5 params with gradients
                        grad_mean = param.grad.abs().mean().item()
                        param_mean = param.data.abs().mean().item()
                        fps_param_samples.append((name.split('.')[-2:], param_mean, grad_mean))
                else:
                    fps_params_without_grad += 1
        
        # Debug output removed - FPS gradient tracking no longer needed
                
        return
        
        # Track FPS conditioning MLP
        fps_cond = model.transformer.fps_conditioning
        if hasattr(fps_cond, 'lin1'):
            w1_mean = fps_cond.lin1.weight.data.abs().mean().item()
            w1_grad = fps_cond.lin1.weight.grad.abs().mean().item() if fps_cond.lin1.weight.grad is not None else 0.0
            print(f"  FPS MLP lin1.weight: mean={w1_mean:.6f}, grad_mean={w1_grad:.6f}")
            
        if hasattr(fps_cond, 'lin2'):
            w2_mean = fps_cond.lin2.weight.data.abs().mean().item()
            w2_grad = fps_cond.lin2.weight.grad.abs().mean().item() if fps_cond.lin2.weight.grad is not None else 0.0
            print(f"  FPS MLP lin2.weight: mean={w2_mean:.6f}, grad_mean={w2_grad:.6f}")
        
        # Track a few FPS adapters
        sample_blocks = [27, 35, 39]  # Sample blocks
        for block_idx in sample_blocks:
            if block_idx < len(model.transformer.blocks):
                block = model.transformer.blocks[block_idx]
                if hasattr(block, 'fps_adapter') and block.fps_adapter is not None:
                    adapter = block.fps_adapter
                    
                    # Gate alpha
                    gate_val = adapter.gate_alpha.item()
                    gate_grad = adapter.gate_alpha.grad.item() if adapter.gate_alpha.grad is not None else 0.0
                    print(f"  Block {block_idx} gate_alpha: val={gate_val:.4f}, grad={gate_grad:.6f}")
                    
                    # LoRA weights
                    k_down_mean = adapter.k_fps_down.weight.data.abs().mean().item()
                    k_down_grad = adapter.k_fps_down.weight.grad.abs().mean().item() if adapter.k_fps_down.weight.grad is not None else 0.0
                    
                    k_up_mean = adapter.k_fps_up.weight.data.abs().mean().item()
                    k_up_grad = adapter.k_fps_up.weight.grad.abs().mean().item() if adapter.k_fps_up.weight.grad is not None else 0.0
                    
                    print(f"  Block {block_idx} k_fps_down: mean={k_down_mean:.6f}, grad={k_down_grad:.6f}")
                    print(f"  Block {block_idx} k_fps_up: mean={k_up_mean:.6f}, grad={k_up_grad:.6f}")
    
    # Log initial FPS parameter magnitudes before training starts
    log_fps_param_magnitudes(model_engine, 0)
    
    epoch_loss = 0
    num_steps = 0
    empty_cuda_cache()
    while True:
        model_engine.reset_activation_shape()
        iterator = get_data_iterator_for_step(train_dataloader, model_engine)
        loss = model_engine.train_batch(iterator).item()
        epoch_loss += loss
        num_steps += 1
        
        # Debug: Track FPS parameter updates every few steps
        if step % 2 == 0:  # Log every 2 steps for more visibility
            debug_fps_param_updates(model_engine, step)
        
        train_dataloader.sync_epoch()

        new_epoch, checkpointed, saved = saver.process_epoch(epoch, step)
        finished_epoch = True if new_epoch != epoch else False

        if is_main_process() and step % config['logging_steps'] == 0:
            tb_writer.add_scalar(f'train/loss', loss, step)
            if wandb_enable:
                wandb.log({'train/loss': loss, 'step': step})
            if optimizer.__class__.__name__ == 'Prodigy':
                prodigy_d = get_prodigy_d(optimizer)
                tb_writer.add_scalar(f'train/prodigy_d', prodigy_d, step)
            if optimizer.__class__.__name__ == 'Automagic':
                lrs, avg_lr = _get_automagic_lrs(optimizer)
                tb_writer.add_histogram(f'train/automagic_lrs', lrs, step)
                tb_writer.add_scalar(f'train/automagic_avg_lr', avg_lr, step)

        if (config['eval_every_n_steps'] and step % config['eval_every_n_steps'] == 0) or (finished_epoch and config['eval_every_n_epochs'] and epoch % config['eval_every_n_epochs'] == 0):
            evaluate(model, model_engine, eval_dataloaders, tb_writer, step, config['eval_gradient_accumulation_steps'], disable_block_swap_for_eval)

        if finished_epoch:
            if is_main_process():
                tb_writer.add_scalar(f'train/epoch_loss', epoch_loss/num_steps, epoch)
                if wandb_enable:
                    wandb.log({'train/epoch_loss': epoch_loss/num_steps, 'epoch': epoch})
            
            # Log FPS parameter magnitudes at end of epoch
            log_fps_param_magnitudes(model_engine, epoch)
            
            epoch_loss = 0
            num_steps = 0
            epoch = new_epoch
            if epoch is None:
                break

        saver.process_step(step)
        step += 1

    # Save final training state checkpoint and model, unless we just saved them.
    if not checkpointed:
        saver.save_checkpoint(step)
    if not saved:
        saver.save_model(f'epoch{epoch}')

    if is_main_process():
        print('TRAINING COMPLETE!')
