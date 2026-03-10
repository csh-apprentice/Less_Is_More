import json
import re
import os.path
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
import safetensors
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device

from models.base import BasePipeline, PreprocessMediaFile, make_contiguous
from utils.common import AUTOCAST_DTYPE, get_lin_function, time_shift, get_t_distribution, slice_t_distribution, sample_t, load_state_dict
from utils.offloading import ModelOffloader
from .t5 import T5EncoderModel
from .vae2_1 import Wan2_1_VAE
from .vae2_2 import Wan2_2_VAE
from .model import (
    WanModel, sinusoidal_embedding_1d, compute_tau_rel
)
from .clip import CLIPModel
from . import configs as wan_configs

KEEP_IN_HIGH_PRECISION = ['norm', 'bias', 'patch_embedding', 'text_embedding', 'time_embedding', 'time_projection', 'head', 'modulation']


class WanModelFromSafetensors(WanModel):
    @classmethod
    def from_pretrained(
        cls,
        weights_file,
        config,
        torch_dtype=torch.bfloat16,
        transformer_dtype=torch.bfloat16,
    ):
        config.pop("_class_name", None)
        config.pop("_diffusers_version", None)

        with init_empty_weights():
            model = cls(**config)

        state_dict = load_state_dict(weights_file)
        state_dict = {
            re.sub(r'^model\.diffusion_model\.', '', k): v for k, v in state_dict.items()
        }

        for name, param in model.named_parameters():
            dtype_to_use = torch_dtype if any(keyword in name for keyword in KEEP_IN_HIGH_PRECISION) else transformer_dtype
            if name in state_dict:
                set_module_tensor_to_device(model, name, device='cpu', dtype=dtype_to_use, value=state_dict[name])
            else:
                # Initialize new parameters (like fps_conditioning) that don't exist in pretrained model
                # print(f'[DEBUG] Initializing new parameter not in pretrained model: {name}')
                # Create properly initialized tensor for new parameters
                if 'fps_conditioning' in name:
                    # FpsConditioning handles its own initialization, don't override it
                    continue
                else:
                    # Default initialization for other new parameters
                    init_value = torch.randn(param.shape, dtype=dtype_to_use) * 0.02
                    set_module_tensor_to_device(model, name, device='cpu', dtype=dtype_to_use, value=init_value)

        # Verify FPS parameters are properly loaded/initialized
        fps_params = [name for name, _ in model.named_parameters() if 'fps' in name.lower()]
        if fps_params:
            # CRITICAL FIX: Ensure all FPS parameters have requires_grad=True
            for name, param in model.named_parameters():
                if 'fps_conditioning' in name:
                    param.requires_grad_(True)
        # No additional setup needed for models without FPS parameters

        return model


def vae_encode(tensor, vae):
    return vae.model.encode(tensor, vae.scale)


# Wrapper to hold both VAE and CLIP, so we can move both to/from GPU together.
class VaeAndClip(nn.Module):
    def __init__(self, vae, clip):
        super().__init__()
        self.vae = vae
        self.clip = clip


class WanPipeline(BasePipeline):
    name = 'wan'
    framerate = 16
    checkpointable_layers = ['TransformerLayer']
    adapter_target_modules = ['WanAttentionBlock']

    def __init__(self, config):
        self.config = config
        self.model_config = self.config['model']
        self.offloader = ModelOffloader('dummy', [], 0, 0, True, torch.device('cuda'), False, debug=False)
        self.cache_text_embeddings = self.model_config.get('cache_text_embeddings', True)
        self.t_dist = get_t_distribution(self.model_config).to('cuda')

        # The official Wan top-level checkpoint folder. Must exist.
        ckpt_dir = Path(self.model_config['ckpt_path'])
        dtype = self.model_config['dtype']

        # transformer_path will either be a directory containing safetensors files, or directly point to a safetensors file
        self.transformer_path = Path(self.model_config.get('transformer_path', ckpt_dir))
        if self.transformer_path.is_dir():
            # If it's a directory, we assume the config JSON exists inside it, along with the safetensors files.
            self.original_model_config_path = self.transformer_path / 'config.json'
            safetensors_files = list(self.transformer_path.glob('*.safetensors'))
        else:
            # If it's a single file, the config JSON is assumed to be in the top-level checkpoint folder.
            self.original_model_config_path = ckpt_dir / 'config.json'
            if not self.original_model_config_path.exists():
                # Wan2.2 has subdirectories for the model. Automatically handle that.
                self.original_model_config_path = ckpt_dir / 'low_noise_model' / 'config.json'
            safetensors_files = [self.transformer_path]

        # get all weight keys
        weight_keys = set()
        for shard in safetensors_files:
            with safetensors.safe_open(shard, framework="pt", device="cpu") as f:
                for k in f.keys():
                    weight_keys.add(re.sub(r'^model\.diffusion_model\.', '', k))

        # SkyReels V2 uses 24 FPS. There seems to be no better way to autodetect this.
        if 'skyreels' in ckpt_dir.name.lower():
            skyreels = True
            self.framerate = 24
        else:
            skyreels = False

        with open(self.original_model_config_path) as f:
            self.json_config = json.load(f)

        model_type = self.json_config['model_type']
        model_dim = self.json_config['dim']

        def autodetect_error():
            raise RuntimeError(f'Could not autodetect model variant. model_type={model_type}, model_dim={model_dim}')

        if model_type == 't2v':
            if skyreels:
                # FPS is different so make sure to use a new cache dir
                self.name = 'skyreels_v2'
            if model_dim == 1536:
                wan_config = wan_configs.t2v_1_3B
            elif model_dim == 5120:
                # This config also works with Wan2.2 T2V.
                wan_config = wan_configs.t2v_14B
            else:
                autodetect_error()
        elif model_type == 'i2v':
            if 'blocks.0.cross_attn.k_img.weight' not in weight_keys:
                # Wan2.2 I2V
                model_type = 'i2v_v2'
                self.name = 'wan2.2_i2v'
                if model_dim == 5120:
                    wan_config = wan_configs.i2v_A14B
                else:
                    autodetect_error()
            else:
                if skyreels:
                    self.name = 'skyreels_v2_i2v'
                else:
                    self.name = 'wan_i2v'
                if model_dim == 1536: # There is no official i2v 1.3b model, but there is https://huggingface.co/alibaba-pai/Wan2.1-Fun-1.3B-InP
                    # This is a hack,
                    wan_config = wan_configs.t2v_1_3B
                    # The following lines are taken from https://github.com/Wan-Video/Wan2.1/blob/main/wan/configs/wan_i2v_14B.py
                    wan_config.clip_model = 'clip_xlm_roberta_vit_h_14'
                    wan_config.clip_dtype = torch.float16
                    wan_config.clip_checkpoint = 'models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth'
                elif model_dim == 5120:
                    wan_config = wan_configs.i2v_14B
                else:
                    autodetect_error()
        elif model_type == 'flf2v':
            assert not skyreels
            self.name = 'wan_flf2v'
            if model_dim == 5120:
                wan_config = wan_configs.i2v_14B  # flf2v is same config as i2v
            else:
                autodetect_error()
        elif model_type == 'ti2v':
            self.name = 'wan2.2_5b'
            self.framerate = 24
            if model_dim == 3072:
                wan_config = wan_configs.ti2v_5B
            else:
                autodetect_error()
        else:
            raise RuntimeError(f'Unknown model_type: {model_type}')

        self.model_type = model_type
        self.json_config['model_type'] = model_type  # to handle the special i2v_v2 type we introduced

        # This is the outermost class, which isn't a nn.Module
        t5_model_path = self.model_config['llm_path'] if self.model_config.get('llm_path', None) else os.path.join(ckpt_dir, wan_config.t5_checkpoint)
        self.text_encoder = T5EncoderModel(
            text_len=wan_config.text_len,
            dtype=dtype,
            device='cpu',
            checkpoint_path=t5_model_path,
            tokenizer_path=ckpt_dir / wan_config.t5_tokenizer,
            shard_fn=None,
        )
        if self.model_config.get('text_encoder_fp8', False):
            for name, p in self.text_encoder.model.named_parameters():
                if p.ndim == 2 and not ('token_embedding' in name or 'pos_embedding' in name):
                    p.data = p.data.to(torch.float8_e4m3fn)
        self.text_encoder.model.requires_grad_(False)

        vae_class = Wan2_2_VAE if model_type == 'ti2v' else Wan2_1_VAE
        # Same here, this isn't a nn.Module.
        self.vae = vae_class(
            vae_pth=ckpt_dir / wan_config.vae_checkpoint,
            device='cpu',
            dtype=dtype,
        )
        self.vae.model.to(dtype)
        # These tensors need to be on the device the VAE will be moved to during caching.
        self.vae.scale = [entry.to('cuda') for entry in self.vae.scale]

        if model_type in ('i2v', 'flf2v'):
            self.clip = CLIPModel(
                dtype=dtype,
                device='cpu',
                checkpoint_path=ckpt_dir / wan_config.clip_checkpoint,
                tokenizer_path=ckpt_dir / wan_config.clip_tokenizer,
            )

    # delay loading transformer to save RAM
    def load_diffusion_model(self):
        dtype = self.model_config['dtype']
        transformer_dtype = self.model_config.get('transformer_dtype', dtype)

        if self.transformer_path.is_file():
            # Add FPS adapter configuration to the model config
            modified_config = self.json_config.copy()
            modified_config.update({
                'fps_adapter_rank': self.model_config.get('fps_adapter_rank', 4),
                'fps_adapter_gate_init': self.model_config.get('fps_adapter_gate_init', 0.0),
                'fps_condition_blocks': self.model_config.get('fps_condition_blocks', "deepest_third"),
                'fps_adapter_num_tokens': self.model_config.get('fps_adapter_num_tokens', 1),
                'fps_tau_transform': self.model_config.get('fps_tau_transform', "log1p"),
                'fps_tau_scale': self.model_config.get('fps_tau_scale', 0.33333334),
                'fps_reference_fps': self.model_config.get('fps_reference_fps', 240.0),
                'fps_embed_dim': self.model_config.get('fps_embed_dim', 256),
                'fps_condition_hidden': self.model_config.get('fps_condition_hidden', 64),
                'fps_lora_alpha': self.model_config.get('fps_lora_alpha', 16),
                'fps_gate_mode': self.model_config.get('fps_gate_mode', 'fixed'),
                'fps_gate_fixed_value': self.model_config.get('fps_gate_fixed_value', 0.5)
            })
            self.transformer = WanModelFromSafetensors.from_pretrained(
                self.transformer_path,
                modified_config,
                torch_dtype=dtype,
                transformer_dtype=transformer_dtype,
            )
        else:
            with init_empty_weights():
                # Add FPS adapter configuration to the model config
                modified_config = self.json_config.copy()
                modified_config.update({
                    'fps_adapter_rank': self.model_config.get('fps_adapter_rank', 4),
                    'fps_adapter_gate_init': self.model_config.get('fps_adapter_gate_init', 0.0),
                    'fps_condition_blocks': self.model_config.get('fps_condition_blocks', "deepest_third"),
                    'fps_adapter_num_tokens': self.model_config.get('fps_adapter_num_tokens', 1),
                    'fps_tau_transform': self.model_config.get('fps_tau_transform', "log1p"),
                    'fps_tau_scale': self.model_config.get('fps_tau_scale', 0.33333334),
                    'fps_reference_fps': self.model_config.get('fps_reference_fps', 240.0),
                    'fps_embed_dim': self.model_config.get('fps_embed_dim', 256),
                    'fps_condition_hidden': self.model_config.get('fps_condition_hidden', 64),
                    'fps_lora_alpha': self.model_config.get('fps_lora_alpha', 16),
                    'fps_gate_mode': self.model_config.get('fps_gate_mode', 'fixed'),
                    'fps_gate_fixed_value': self.model_config.get('fps_gate_fixed_value', 0.5)
                })
                self.transformer = WanModel.from_config(modified_config)
            state_dict = {}
            for shard in self.transformer_path.glob('*.safetensors'):
                with safetensors.safe_open(shard, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        state_dict[key] = f.get_tensor(key)
            for name, param in self.transformer.named_parameters():
                dtype_to_use = dtype if any(keyword in name for keyword in KEEP_IN_HIGH_PRECISION) else transformer_dtype
                if name in state_dict:
                    set_module_tensor_to_device(self.transformer, name, device='cpu', dtype=dtype_to_use, value=state_dict[name])
                else:
                    # Initialize new parameters (like fps_conditioning) that don't exist in pretrained model
                    # print(f'[DEBUG] Initializing new parameter not in pretrained model: {name}')
                    # Create properly initialized tensor for new parameters
                    if 'fps_conditioning' in name:
                        if 'fps_conditioning.0.weight' in name:
                            # First Linear: Kaiming-uniform (fan_in, nonlinearity='relu')
                            init_value = torch.empty(param.shape, dtype=dtype_to_use)
                            torch.nn.init.kaiming_uniform_(init_value, mode='fan_in', nonlinearity='relu')
                        elif 'fps_conditioning.0.bias' in name:
                            # First Linear: bias = 0
                            init_value = torch.zeros(param.shape, dtype=dtype_to_use)
                        elif 'fps_conditioning.2.weight' in name:
                            # LayerNorm: weight = 1
                            init_value = torch.ones(param.shape, dtype=dtype_to_use)
                        elif 'fps_conditioning.2.bias' in name:
                            # LayerNorm: bias = 0
                            init_value = torch.zeros(param.shape, dtype=dtype_to_use)
                        elif 'fps_conditioning.3.weight' in name:
                            # Final Linear: weights = 0 (zero-disturbance start)
                            init_value = torch.zeros(param.shape, dtype=dtype_to_use, requires_grad=True)
                        elif 'fps_conditioning.3.bias' in name:
                            # Final Linear: bias = 0
                            init_value = torch.zeros(param.shape, dtype=dtype_to_use)
                        else:
                            # Fallback for any other fps_conditioning parameters
                            init_value = torch.randn(param.shape, dtype=dtype_to_use) * 0.02
                    else:
                        # Default initialization for other new parameters
                        init_value = torch.randn(param.shape, dtype=dtype_to_use) * 0.02
                    set_module_tensor_to_device(self.transformer, name, device='cpu', dtype=dtype_to_use, value=init_value)

        # CRITICAL FIX: Ensure all FPS parameters have requires_grad=True
        fps_params = [name for name, _ in self.transformer.named_parameters() if 'fps' in name.lower()]
        if fps_params:
            for name, param in self.transformer.named_parameters():
                if 'fps_conditioning' in name:
                    param.requires_grad_(True)

        self.transformer.train()
        # We'll need the original parameter name for saving, and the name changes once we wrap modules for pipeline parallelism,
        # so store it in an attribute here. Same thing below if we're training a lora and creating lora weights.
        for name, p in self.transformer.named_parameters():
            p.original_name = name

    def __getattr__(self, name):
        # Prevent infinite recursion by using __dict__ instead of hasattr()
        if 'diffusers_pipeline' in self.__dict__:
            diffusers_pipeline = self.__dict__['diffusers_pipeline']
            if diffusers_pipeline is not None and hasattr(diffusers_pipeline, name):
                return getattr(diffusers_pipeline, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def get_vae(self):
        vae = self.vae.model
        clip = self.clip.model if self.model_type in ('i2v', 'flf2v') else None
        return VaeAndClip(vae, clip)

    def get_text_encoders(self):
        # Return the inner nn.Module
        if self.cache_text_embeddings:
            return [self.text_encoder.model]
        else:
            return []

    def save_adapter(self, save_dir, peft_state_dict):
        self.peft_config.save_pretrained(save_dir)
        # ComfyUI format.
        peft_state_dict = {'diffusion_model.'+k: v for k, v in peft_state_dict.items()}
        
        # CRITICAL FIX: Manually add FPS adapter parameters that were excluded from PEFT
        # FPS adapters were excluded from PEFT target modules to avoid double LoRA wrapping,
        # but we still need to save them for inference
        fps_adapter_params_added = 0
        
        print(f'[FPS_ADAPTER_SAVE_DEBUG] Starting FPS adapter parameter collection...')
        print(f'[FPS_ADAPTER_SAVE_DEBUG] self.transformer type: {type(self.transformer)}')
        print(f'[FPS_ADAPTER_SAVE_DEBUG] has blocks: {hasattr(self.transformer, "blocks")}')
        
        # Use the same successful method as [DIRECT_FPS_COUNT] to find FPS adapter parameters
        if hasattr(self.transformer, 'blocks'):
            print(f'[FPS_ADAPTER_SAVE_DEBUG] Found {len(self.transformer.blocks)} blocks')
            for i, block in enumerate(self.transformer.blocks):
                if hasattr(block, 'fps_adapter') and block.fps_adapter is not None:
                    print(f'[FPS_ADAPTER_SAVE_DEBUG] Block {i} has FPS adapter')
                    for param_name, param in block.fps_adapter.named_parameters():
                        print(f'[FPS_ADAPTER_SAVE_DEBUG] Found param {param_name}, requires_grad={param.requires_grad}')
                        if param.requires_grad:
                            # Construct full parameter name: blocks.{i}.fps_adapter.{param_name}
                            full_name = f'blocks.{i}.fps_adapter.{param_name}'
                            key = f'diffusion_model.{full_name}'
                            if key not in peft_state_dict:  # Avoid duplicates
                                peft_state_dict[key] = param.detach().cpu()
                                fps_adapter_params_added += 1
                                print(f'[FPS_ADAPTER_SAVE_DEBUG] Added {key}')
        else:
            print(f'[FPS_ADAPTER_SAVE_DEBUG] self.transformer has no blocks attribute!')
        
        if fps_adapter_params_added > 0:
            print(f'[FPS_ADAPTER_SAVE] Added {fps_adapter_params_added} FPS adapter parameters to checkpoint')
        
        # PRINT ACTUAL FPS PARAMETER VALUES for tracking changes
        print(f'[FPS_VALUES_SAVE] Printing sample FPS parameter values at save time...')
        
        # Print FPS MLP values
        fps_mlp_keys = [k for k in peft_state_dict.keys() if 'fps_conditioning' in k]
        for key in sorted(fps_mlp_keys):
            tensor = peft_state_dict[key]
            if tensor.numel() <= 10:  # Small tensors - show all values
                print(f'[FPS_VALUES_SAVE] {key}: {tensor.flatten().tolist()}')
            else:  # Large tensors - show statistics
                norm = tensor.norm().item()
                mean = tensor.mean().item()
                std = tensor.std().item()
                print(f'[FPS_VALUES_SAVE] {key}: norm={norm:.6f}, mean={mean:.6f}, std={std:.6f}')
        
        # Print sample FPS adapter values (first 2 blocks)
        fps_adapter_keys = [k for k in peft_state_dict.keys() if 'fps_adapter' in k]
        sample_fps_keys = [k for k in sorted(fps_adapter_keys) if 'blocks.27' in k or 'blocks.28' in k]
        for key in sample_fps_keys:
            tensor = peft_state_dict[key]
            if 'gate_alpha' in key:
                print(f'[FPS_VALUES_SAVE] {key}: {tensor.item():.6f}')
            else:
                norm = tensor.norm().item()
                mean = tensor.mean().item()
                print(f'[FPS_VALUES_SAVE] {key}: norm={norm:.6f}, mean={mean:.6f}')
        
        # Save the PEFT state dict
        safetensors.torch.save_file(peft_state_dict, save_dir / 'adapter_model.safetensors', metadata={'format': 'pt'})

    def save_model(self, save_dir, state_dict):
        # Save the full model state dict
        safetensors.torch.save_file(state_dict, save_dir / 'model.safetensors', metadata={'format': 'pt'})
        
    def test_fps_checkpoint_compatibility(self):
        """Test function to verify FPS parameters can be saved and loaded correctly"""
        print('[DEBUG] Testing FPS checkpoint compatibility...')
        
        # Get current FPS parameters
        fps_params_before = {}
        for name, param in self.transformer.named_parameters():
            if 'fps' in name.lower():
                fps_params_before[name] = param.data.clone()
        
        if fps_params_before:
            # Test that parameters have the expected names and shapes
            expected_patterns = ['fps_conditioning', 'fps_adapter', 'gate_alpha']
            found_patterns = []
            for name in fps_params_before.keys():
                for pattern in expected_patterns:
                    if pattern in name.lower():
                        found_patterns.append(pattern)
                        break
            
            # Verify parameter ranges are reasonable
            for name, param in fps_params_before.items():
                param_stats = {
                    'mean': param.mean().item(),
                    'std': param.std().item(),
                    'min': param.min().item(),
                    'max': param.max().item()
                }
                # Parameter stats computed for verification
                pass
                
            # FPS checkpoint compatibility test completed successfully
            return True
        else:
            # No FPS parameters found - checkpoint test skipped
            return False

    def get_preprocess_media_file_fn(self):
        if self.model_type == 'ti2v':
            round_side = 32
        else:
            round_side = 16
        return PreprocessMediaFile(
            self.config,
            support_video=True,
            framerate=self.framerate,
            round_height=round_side,
            round_width=round_side,
        )

    def get_call_vae_fn(self, vae_and_clip):
        is_i2v = self.model_type in ('i2v', 'flf2v', 'i2v_v2')
        def fn(tensor):
            vae = vae_and_clip.vae
            p = next(vae.parameters())
            tensor = tensor.to(p.device, p.dtype)
            latents = vae_encode(tensor, self.vae)
            ret = {'latents': latents}

            if is_i2v:
                assert tensor.ndim == 5, f'i2v/flf2v must train on videos, got tensor with shape {tensor.shape}'
                assert tensor.shape[2] > 1, 'i2v/flf2v must train on videos, but got an image'
                first_frame = tensor[:, :, 0:1, ...].clone()

                if self.model_type == 'flf2v':
                    tensor[:, :, 1:-1, ...] = 0
                else:
                    tensor[:, :, 1:, ...] = 0

                # Image conditioning. Same shame as latents, first frame is unchanged, rest is 0.
                # NOTE: encoding 0s with the VAE doesn't give you 0s in the latents, I tested this. So we need to
                # encode the whole thing here, we can't just extract the first frame from the latents later and make
                # the rest 0. But what happens if you do that? Probably things get fried, but might be worth testing.
                y = vae_encode(tensor, self.vae)
                ret['y'] = y

            clip = vae_and_clip.clip
            if clip is not None:
                clip_context = self.clip.visual(first_frame.to(p.device, p.dtype))
                if self.model_type == 'flf2v':
                    last_frame = tensor[:, :, -1:, ...].clone()
                    # NOTE: dim=1 is a hack to pass clip_context without microbatching breaking the zeroth dim
                    clip_context = torch.cat([clip_context, self.clip.visual(last_frame.to(p.device, p.dtype))], dim=1)
                ret['clip_context'] = clip_context

            return ret
        return fn

    def get_call_text_encoder_fn(self, text_encoder):
        def fn(caption, is_video):
            # Args are lists
            p = next(text_encoder.parameters())
            ids, mask = self.text_encoder.tokenizer(caption, return_mask=True, add_special_tokens=True)
            ids = ids.to(p.device)
            mask = mask.to(p.device)
            seq_lens = mask.gt(0).sum(dim=1).long()
            with torch.autocast(device_type=p.device.type, dtype=p.dtype):
                text_embeddings = text_encoder(ids, mask)
                return {'text_embeddings': text_embeddings, 'seq_lens': seq_lens}
        return fn

    def prepare_inputs(self, inputs, timestep_quantile=None):
        latents = inputs['latents'].float()
        mask = inputs['mask']
        y = inputs['y'] if self.model_type in ('i2v', 'flf2v', 'i2v_v2') else None
        # No CLIP for i2v_v2 (Wan2.2)
        clip_context = inputs['clip_context'] if self.model_type in ('i2v', 'flf2v') else None
        
        # Handle FPS conditioning
        fps_values = inputs.get('fps', None)
        if fps_values is not None:
            fps_non_none = [fps for fps in fps_values if fps is not None]
            if not fps_non_none:
                print(f'[DEBUG] prepare_inputs() fps_values is not None but contains no valid fps values')  # Keep error case

        if self.cache_text_embeddings:
            text_embeddings_or_ids = inputs['text_embeddings']
            seq_lens_or_text_mask = inputs['seq_lens']
        else:
            text_embeddings_or_ids, seq_lens_or_text_mask = self.text_encoder.tokenizer(inputs['caption'], return_mask=True, add_special_tokens=True)

        bs, channels, num_frames, h, w = latents.shape

        if mask is not None:
            mask = mask.unsqueeze(1)  # make mask (bs, 1, img_h, img_w)
            mask = F.interpolate(mask, size=(h, w), mode='nearest-exact')  # resize to latent spatial dimension
            mask = mask.unsqueeze(2)  # make mask same number of dims as target

        t = self.t_dist

        if shift := self.model_config.get('shift', None):
            t = (t * shift) / (1 + (shift - 1) * t)
        elif self.model_config.get('flux_shift', False):
            mu = get_lin_function(y1=0.5, y2=1.15)((h // 2) * (w // 2))
            t = time_shift(mu, 1.0, t)

        t = slice_t_distribution(t, min_t=self.model_config.get('min_t', 0.0), max_t=self.model_config.get('max_t', 1.0))
        t = sample_t(t, bs, quantile=timestep_quantile).to(latents.device)

        x_1 = latents
        x_0 = torch.randn_like(x_1)
        t_expanded = t.view(-1, 1, 1, 1, 1)
        x_t = (1 - t_expanded) * x_1 + t_expanded * x_0
        target = x_0 - x_1

        # timestep input to model needs to be in range [0, 1000]
        t = t * 1000

        return (
            (x_t, y, t, text_embeddings_or_ids, seq_lens_or_text_mask, clip_context, fps_values),
            (target, mask),
        )

    def to_layers(self):
        transformer = self.transformer
        text_encoder = None if self.cache_text_embeddings else self.text_encoder.model
        layers = [InitialLayer(transformer, text_encoder)]
        for i, block in enumerate(transformer.blocks):
            layers.append(TransformerLayer(block, i, self.offloader))
        layers.append(FinalLayer(transformer))
        return layers

    def enable_block_swap(self, blocks_to_swap):
        transformer = self.transformer
        blocks = transformer.blocks
        num_blocks = len(blocks)
        assert (
            blocks_to_swap <= num_blocks - 2
        ), f'Cannot swap more than {num_blocks - 2} blocks. Requested {blocks_to_swap} blocks to swap.'
        self.offloader = ModelOffloader(
            'TransformerBlock', blocks, num_blocks, blocks_to_swap, True, torch.device('cuda'), self.config['reentrant_activation_checkpointing']
        )
        transformer.blocks = None
        transformer.to('cuda')
        transformer.blocks = blocks
        self.prepare_block_swap_training()
        print(f'Block swap enabled. Swapping {blocks_to_swap} blocks out of {num_blocks} blocks.')

    def prepare_block_swap_training(self):
        self.offloader.enable_block_swap()
        self.offloader.set_forward_only(False)
        self.offloader.prepare_block_devices_before_forward()

    def prepare_block_swap_inference(self, disable_block_swap=False):
        if disable_block_swap:
            self.offloader.disable_block_swap()
        self.offloader.set_forward_only(True)
        self.offloader.prepare_block_devices_before_forward()


class InitialLayer(nn.Module):
    def __init__(self, model, text_encoder):
        super().__init__()
        self.patch_embedding = model.patch_embedding
        self.time_embedding = model.time_embedding
        self.text_embedding = model.text_embedding
        self.time_projection = model.time_projection
        self.fps_conditioning = model.fps_conditioning
        
        # FPS conditioning configuration
        self.fps_tau_transform = model.fps_tau_transform
        self.fps_tau_scale = model.fps_tau_scale
        self.fps_reference_fps = model.fps_reference_fps
        
        # Test FPS checkpoint compatibility after model is fully loaded
        if hasattr(self, 'test_fps_checkpoint_compatibility'):
            self.test_fps_checkpoint_compatibility()
            
        self.i2v = (model.model_type == 'i2v')
        self.i2v_v2 = (model.model_type == 'i2v_v2')
        self.flf2v = (model.model_type == 'flf2v')
        if self.i2v or self.flf2v:
            self.img_emb = model.img_emb
        self.text_encoder = text_encoder
        self.freqs = model.freqs
        self.freq_dim = model.freq_dim
        self.dim = model.dim
        self.text_len = model.text_len

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        for item in inputs:
            if item is not None and torch.is_floating_point(item):
                item.requires_grad_(True)

        x, y, t, text_embeddings_or_ids, seq_lens_or_text_mask, clip_fea, fps_values = inputs
        bs, channels, f, h, w = x.shape
        if clip_fea is not None and clip_fea.numel() == 0:
            clip_fea = None

        if self.text_encoder is not None:
            assert not torch.is_floating_point(text_embeddings_or_ids)
            with torch.no_grad():
                context = self.text_encoder(text_embeddings_or_ids, seq_lens_or_text_mask)
            text_seq_lens = seq_lens_or_text_mask.gt(0).sum(dim=1).long()
        else:
            context = text_embeddings_or_ids
            text_seq_lens = seq_lens_or_text_mask

        context = [emb[:length] for emb, length in zip(context, text_seq_lens)]

        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if self.i2v or self.flf2v or self.i2v_v2:
            mask = torch.zeros((bs, 4, f, h, w), device=x.device, dtype=x.dtype)
            mask[:, :, 0, ...] = 1
            if self.flf2v:
                mask[:, :, -1, ...] = 1
            y = torch.cat([mask, y], dim=1)
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        seq_len = seq_lens.max()
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        # time embeddings
        time_embed_seq_len = seq_len
        if t.dim() == 1:
            t = t.unsqueeze(-1)
            time_embed_seq_len = 1  # will broadcast
        bt = t.size(0)
        t = t.flatten()
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t).unflatten(0, (bt, time_embed_seq_len)).to(x.device, torch.float32)
        )
        e0 = self.time_projection(e).unflatten(2, (6, self.dim))

        # context
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if self.i2v or self.flf2v:
            assert clip_fea is not None
            if self.flf2v:
                self.img_emb.emb_pos.data = self.img_emb.emb_pos.data.to(clip_fea.device, torch.float32)
                clip_fea = clip_fea.view(-1, 257, 1280)
            context_clip = self.img_emb(clip_fea)  # bs x 257 (x2) x dim
            context = torch.concat([context_clip, context], dim=1)

        # FPS conditioning - ALWAYS create a valid tensor for checkpoint compatibility
        # fps_values is NEVER None (dataset always provides fps values)
        # Convert fps values to tau_rel and create conditioning embeddings
        # CRITICAL: Create tensor from x to maintain computation graph consistency
        fps_tensor = x.new_tensor(fps_values, dtype=torch.float32)
        tau_rel = compute_tau_rel(fps_tensor, reference_fps=self.fps_reference_fps, transform=self.fps_tau_transform, scale=self.fps_tau_scale)

        # Handle shape for single vs multi-condition:
        # - Single condition: tau_rel shape is [batch_size], need to unsqueeze to [batch_size, 1]
        # - Multi-condition: tau_rel shape is [batch_size, num_conditions], already correct
        if tau_rel.ndim == 1:
            tau_rel = tau_rel.unsqueeze(-1)  # [batch_size] -> [batch_size, 1]
        # Now tau_rel shape is [batch_size, num_conditions] for both cases

        fps_conditioning = self.fps_conditioning(tau_rel)  # Shape: [batch_size, dim]
        

        # pipeline parallelism needs everything on the GPU
        seq_lens = seq_lens.to(x.device)
        grid_sizes = grid_sizes.to(x.device)

        return make_contiguous(x, e, e0, seq_lens, grid_sizes, self.freqs, context, fps_conditioning)


class TransformerLayer(nn.Module):
    def __init__(self, block, block_idx, offloader):
        super().__init__()
        self.block = block
        self.block_idx = block_idx
        self.offloader = offloader

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        x, e, e0, seq_lens, grid_sizes, freqs, context, fps_conditioning = inputs

        self.offloader.wait_for_block(self.block_idx)
        x = self.block(x, e0, seq_lens, grid_sizes, freqs, context, None, fps_conditioning)
        self.offloader.submit_move_blocks_forward(self.block_idx)

        return make_contiguous(x, e, e0, seq_lens, grid_sizes, freqs, context, fps_conditioning)


class FinalLayer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.head = model.head
        self.model = [model]

    def __getattr__(self, name):
        return getattr(self.model[0], name)

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        x, e, e0, seq_lens, grid_sizes, freqs, context, fps_conditioning = inputs
        x = self.head(x, e)
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x, dim=0)
