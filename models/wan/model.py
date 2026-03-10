# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math

import torch
import torch.nn as nn
import torch.nn.init as init
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from .attention import flash_attention

T5_CONTEXT_TOKEN_NUMBER = 512
FIRST_LAST_FRAME_CONTEXT_TOKEN_NUMBER = 257 * 2

from typing import Optional, Union

Number = Union[float, int]
TensorOrNumber = Union[torch.Tensor, Number]

def compute_tau_rel(
    fps: TensorOrNumber,
    *,
    # If you don't want a reference, pass reference_fps=None (uses tau = 1/fps)
    # NEW: Can be a list for multi-condition
    reference_fps: Optional[Union[Number, list]] = 240.0,
    # Transform to apply to tau (exposure proxy)
    # NEW: Can be a list for multi-condition
    transform: Union[str, list] = "log1p",          # options: "raw" | "log" | "log1p" | "neglogfps" | "centerlog1p"
    # Multiplicative scaling (kept as a multiplier, as you prefer)
    # NEW: Can be a list for multi-condition
    scale: Union[float, list] = 0.33333334,         # ≈ 1/3 → keeps log1p(240/fps) ~ [0, ~3] → [0, ~1]
    # Numerical stability
    eps: float = 1e-6,
) -> TensorOrNumber:
    """
    Compute scalar(s) for fps-conditioning.

    NEW: Supports multi-condition input where fps can be a tensor with shape [batch_size, num_conditions].
    In this case, transform, scale, and reference_fps can be lists with length=num_conditions.

    tau_raw = (reference_fps / fps) if reference_fps is not None else (1 / fps)

    transform:
      - "raw":      tau_raw
      - "log":      log(tau_raw + eps)
      - "log1p":    log1p(tau_raw)              # recommended
      - "neglogfps": -log(fps + eps)            # equivalent to log(1/fps) up to a constant
      - "centerlog1p": sign(reference_fps-fps) * log1p(abs(tau-1))  # centered around tau=1

    Returns same type as input (torch.Tensor or float/int).
    For multi-condition, returns tensor with shape [batch_size, num_conditions].
    """
    is_tensor = isinstance(fps, torch.Tensor)
    x = fps.to(torch.float32) if is_tensor else torch.tensor(float(fps), dtype=torch.float32)

    # Check if this is multi-condition (fps has 2 dimensions: [batch, num_conditions])
    is_multi_condition = is_tensor and x.ndim == 2

    if not is_multi_condition:
        # Single condition - original behavior (backward compatible)
        # Ensure transform, scale, reference_fps are scalars
        if isinstance(transform, list):
            transform = transform[0]
        if isinstance(scale, list):
            scale = scale[0]
        if isinstance(reference_fps, list):
            reference_fps = reference_fps[0]

        # transform
        if transform == "raw":
            # Raw: return FPS value directly, no transformation
            y = x
        elif transform == "neglogfps":
            # neglogfps: -log(fps), no reference needed
            y = -torch.log(x + eps)
        else:
            # For other transforms, compute tau first
            if reference_fps is None:
                tau = 1.0 / (x + eps)
            else:
                tau = float(reference_fps) / (x + eps)

            if transform == "log":
                y = torch.log(tau + eps)
            elif transform == "log1p":
                y = torch.log1p(tau)          # smooth & stable for our τ ∈ {1,2,4,6,10,20}
            elif transform == "centerlog1p":
                # Centered log1p transform: sign(reference_fps-fps) * log1p(abs(tau-1))
                # This centers the transform around tau=1 (when fps=reference_fps)
                if reference_fps is None:
                    raise ValueError("centerlog1p transform requires reference_fps to be specified")
                sign_term = torch.sign(float(reference_fps) - x)
                y = sign_term * torch.log1p(torch.abs(tau - 1.0))
            else:
                raise ValueError(f"Unknown transform: {transform}")

        # multiplicative scaling (your preference)
        if scale is not None:
            y = y * float(scale)

        if is_tensor:
            return y
        return float(y.item())

    else:
        # Multi-condition case: x has shape [batch_size, num_conditions]
        batch_size, num_conditions = x.shape

        # Ensure transform, scale, reference_fps are lists with correct length
        if not isinstance(transform, list):
            transform = [transform] * num_conditions
        if not isinstance(scale, list):
            scale = [scale] * num_conditions
        if not isinstance(reference_fps, list):
            reference_fps = [reference_fps] * num_conditions

        if len(transform) != num_conditions:
            raise ValueError(f"transform list length ({len(transform)}) must match num_conditions ({num_conditions})")
        if len(scale) != num_conditions:
            raise ValueError(f"scale list length ({len(scale)}) must match num_conditions ({num_conditions})")
        if len(reference_fps) != num_conditions:
            raise ValueError(f"reference_fps list length ({len(reference_fps)}) must match num_conditions ({num_conditions})")

        # Process each condition separately
        y_list = []
        for i in range(num_conditions):
            x_i = x[:, i]  # Shape: [batch_size]
            ref_fps_i = reference_fps[i]
            transform_i = transform[i]
            scale_i = scale[i]

            # Apply transform for this condition
            if transform_i == "raw":
                # Raw: return FPS value directly, no transformation
                y_i = x_i
            elif transform_i == "neglogfps":
                # neglogfps: -log(fps), no reference needed
                y_i = -torch.log(x_i + eps)
            else:
                # For other transforms, compute tau first
                if ref_fps_i is None:
                    tau_i = 1.0 / (x_i + eps)
                else:
                    tau_i = float(ref_fps_i) / (x_i + eps)

                if transform_i == "log":
                    y_i = torch.log(tau_i + eps)
                elif transform_i == "log1p":
                    y_i = torch.log1p(tau_i)
                elif transform_i == "centerlog1p":
                    if ref_fps_i is None:
                        raise ValueError("centerlog1p transform requires reference_fps to be specified")
                    sign_term = torch.sign(float(ref_fps_i) - x_i)
                    y_i = sign_term * torch.log1p(torch.abs(tau_i - 1.0))
                else:
                    raise ValueError(f"Unknown transform: {transform_i}")

            # Apply scale for this condition
            if scale_i is not None:
                y_i = y_i * float(scale_i)

            y_list.append(y_i.unsqueeze(1))  # Shape: [batch_size, 1]

        # Concatenate all conditions: [batch_size, num_conditions]
        y = torch.cat(y_list, dim=1)
        return y


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


@torch.amp.autocast('cuda', enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


@torch.amp.autocast('cuda', enabled=False)
def rope_apply(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
                            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).float()


class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return super().forward(x.float()).type_as(x)


class WanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        x = flash_attention(
            q=rope_apply(q, grid_sizes, freqs),
            k=rope_apply(k, grid_sizes, freqs),
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanCrossAttention(WanSelfAttention):

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanI2VCrossAttention(WanSelfAttention):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)

        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        # self.alpha = nn.Parameter(torch.zeros((1, )))
        self.norm_k_img = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        image_context_length = context.shape[1] - T5_CONTEXT_TOKEN_NUMBER
        context_img = context[:, :image_context_length]
        context = context[:, image_context_length:]
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
        v_img = self.v_img(context_img).view(b, -1, n, d)
        img_x = flash_attention(q, k_img, v_img, k_lens=None)
        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        img_x = img_x.flatten(2)
        x = x + img_x
        x = self.o(x)
        return x


class FpsConditioning(nn.Module):
    """
    FPS conditioning module with special initialization for zero-disturbance training.
    SubTask 7: Simplified design matching Wan2.1 time embedding (Linear -> SiLU -> Linear).

    NEW: Supports multi-condition input where num_conditions > 1.
    """
    def __init__(self, out_dim: int, hidden: int = 64, num_conditions: int = 1):
        super().__init__()

        # Layers: Linear(num_conditions, hidden) -> SiLU -> Linear(hidden, out_dim)
        # Removed LayerNorm to prevent weakening of FPS signal at initialization
        # num_conditions defaults to 1 for backward compatibility
        self.lin1 = nn.Linear(num_conditions, hidden)
        self.act = nn.SiLU()
        self.lin2 = nn.Linear(hidden, out_dim)

        # Note: Don't call _init_weights() here - tensors are on meta device
        # Custom initialization will be applied after materialization
    
    def _init_weights(self):
        """Special initialization for zero-disturbance training using PyTorch init functions."""
        # lin1: Kaiming uniform initialization (fan_in, ReLU-like)
        nn.init.kaiming_uniform_(self.lin1.weight, a=0.0, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.lin1.bias)
        
        # lin2: Zero initialization for zero-disturbance head (like LoRA B matrices)
        nn.init.zeros_(self.lin2.weight)
        nn.init.zeros_(self.lin2.bias)
        
    
    def forward(self, x):
        """Forward pass: lin1 -> SiLU -> lin2"""
        return self.lin2(self.act(self.lin1(x)))


class FPSCrossAttentionAdapter(nn.Module):
    """
    Checkpoint-safe FPS conditioning adapter using LoRA projections for disentangled cross-attention.
    Implements: y = Attn(Q, K_text, V_text) + g * Attn(Q, K', V')
    
    IMPORTANT: This version ensures deterministic tensor shapes for gradient checkpointing compatibility.
    """
    
    # Class variable to track one representative adapter for debugging
    _debug_instance = None
    _global_call_count = 0
    _total_adapters_created = 0
    
    def __init__(self, dim, num_heads, fps_conditioning_dim, rank=8, gate_init=0.0, num_tokens=1, lora_alpha=16,
                 gate_mode='sigmoid', gate_fixed_value=0.5, fps_scale=False, warmup_steps=100):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.rank = rank
        self.fps_conditioning_dim = fps_conditioning_dim
        self.num_tokens = num_tokens  # Number of FPS conditioning tokens per head

        # LoRA scaling factor: lora_alpha / rank (standard LoRA scaling)
        self.lora_scale = lora_alpha / rank

        # FPS scaling and warmup options
        self.fps_scale = fps_scale  # Whether to scale y_fps to match y_text magnitude
        self.warmup_steps = warmup_steps  # Number of steps for warm-up schedule
        self.register_buffer('current_step', torch.tensor(0))  # Track training steps for warmup
        
        # LoRA projections for K' and V' - always executed regardless of FPS values
        # Output dimension is num_tokens * dim to support multiple conditioning tokens
        self.k_fps_down = nn.Linear(fps_conditioning_dim, rank, bias=False)
        self.k_fps_up = nn.Linear(rank, num_tokens * dim, bias=False)
        self.v_fps_down = nn.Linear(fps_conditioning_dim, rank, bias=False)
        self.v_fps_up = nn.Linear(rank, num_tokens * dim, bias=False)
        
        # RMSNorm for K matrix to match text K normalization scale
        # This ensures K_fps and K_text are on the same scale before attention
        self.norm_k_fps = WanRMSNorm(dim, eps=1e-6)
        
        # Configurable gate system
        self.gate_mode = gate_mode
        self.gate_init = gate_init  # Store original value for learnable modes
        self.gate_fixed_value = gate_fixed_value  # Store fixed value for fixed mode

        # Supported gate modes: 'sigmoid', 'identity', 'relu', 'silu', 'softplus', 'fixed'
        if gate_mode in ['sigmoid', 'identity', 'relu', 'silu', 'softplus']:
            # Learnable gate parameter (trainable) with different activation functions
            # sigmoid: gate = sigmoid(alpha) ∈ (0, 1)
            # identity: gate = alpha (no activation, can be any value)
            # relu: gate = ReLU(alpha) ∈ [0, ∞)
            # silu: gate = SiLU(alpha) = alpha * sigmoid(alpha)
            # softplus: gate = softplus(alpha) = log(1 + exp(alpha)) ∈ (0, ∞), smooth ReLU
            self.gate_alpha = nn.Parameter(torch.tensor(gate_init))
        elif gate_mode == 'fixed':
            # Fixed gate mode - use constant gate value directly, no trainable parameters
            self.register_buffer('gate_fixed', torch.tensor(gate_fixed_value))
        else:
            raise ValueError(f"Unknown gate_mode: {gate_mode}. Must be one of: 'sigmoid', 'identity', 'relu', 'silu', 'softplus', 'fixed'")
        
        # Debug counter (only print every N forward passes)
        self.debug_counter = 0
        
        # Initialize LoRA weights properly
        self._init_lora_weights()
        
        # Count FPS adapter creation
        FPSCrossAttentionAdapter._total_adapters_created += 1
        
        # Set first instance as debug instance
        if FPSCrossAttentionAdapter._debug_instance is None:
            FPSCrossAttentionAdapter._debug_instance = self
        
    def _init_lora_weights(self):
        """Initialize LoRA weights for FPS adapter with gradient flow fix"""
        # Down projections (A): random init  
        nn.init.normal_(self.k_fps_down.weight, std=0.01)
        nn.init.normal_(self.v_fps_down.weight, std=0.01)
        
        # Up projections (B): SMALL RANDOM INIT (fixes gradient flow)
        # Unlike standard LoRA, we have no base weight matrix, so B=0 would
        # create a dead branch with no gradients. Use very small random init.
        nn.init.normal_(self.k_fps_up.weight, std=1e-4)  # Very small to minimize initial impact
        nn.init.normal_(self.v_fps_up.weight, std=1e-4)  # But non-zero to allow gradients
        
        # Ensure RMSNorm weights are properly initialized to ones (should be default but verify)
        with torch.no_grad():
            self.norm_k_fps.weight.fill_(1.0)
        
        # Re-initialize gate parameters based on gate mode (fixes DeepSpeed override)
        if self.gate_mode in ['sigmoid', 'identity', 'relu', 'silu', 'softplus'] and hasattr(self, 'gate_init'):
            with torch.no_grad():
                self.gate_alpha.fill_(self.gate_init)
        # Note: Fixed gate mode doesn't need reinitialization as it uses a buffer
    
    def verify_initialization(self):
        """Verify FPS adapter initialization is correct (updated for gradient flow fix)"""
        # Only verify if tensors are not meta tensors
        if not self.k_fps_down.weight.is_meta:
            # Check down projections have non-zero weights (random init)
            k_down_nonzero = not torch.all(self.k_fps_down.weight == 0).item()
            v_down_nonzero = not torch.all(self.v_fps_down.weight == 0).item()
            
            # Check up projections are SMALL but NON-ZERO (gradient flow fix)
            k_up_nonzero = not torch.all(self.k_fps_up.weight == 0).item()
            v_up_nonzero = not torch.all(self.v_fps_up.weight == 0).item()
            k_up_small = self.k_fps_up.weight.abs().max().item() < 0.01  # Should be small
            v_up_small = self.v_fps_up.weight.abs().max().item() < 0.01  # Should be small
            
            # Verify initialization is correct
            init_correct = (k_down_nonzero and v_down_nonzero and 
                          k_up_nonzero and v_up_nonzero and
                          k_up_small and v_up_small)
            # Verify gate initialization based on gate mode
            if self.gate_mode in ['sigmoid', 'identity', 'relu', 'silu', 'softplus']:
                gate_correct = abs(self.gate_alpha.item() - self.gate_init) < 1e-5
            elif self.gate_mode == 'fixed':
                gate_correct = abs(self.gate_fixed.item() - self.gate_fixed_value) < 1e-5
            else:
                gate_correct = False
            
            return init_correct and gate_correct
        else:
            # Weights are meta tensors - skip verification
            return None
    
    def _count_adapter_parameters(self):
        """Count parameters in this specific FPS adapter"""
        total_params = 0
        param_details = []
        
        # Count each LoRA matrix
        for name, param in self.named_parameters():
            total_params += param.numel()
            param_details.append(f"{name}:{param.numel()}")
            
        return f"{total_params} total ({', '.join(param_details)})"
    
    @classmethod
    def get_total_adapter_count_and_params(cls):
        """Get total count and parameters across all FPS adapter instances"""
        return cls._total_adapters_created
        
    def forward(self, q, k_text, v_text, fps_conditioning, context_lens):
        """
        Checkpoint-safe forward pass using only deterministic PyTorch operations.
        Always follows identical computation path regardless of fps_conditioning value.
        """
        
        # ALWAYS compute text cross-attention: Attn(Q, K_text, V_text)
        y_text = flash_attention(q, k_text, v_text, k_lens=context_lens)
        
        # 1) project
        k_fps_proj = self.k_fps_up(self.k_fps_down(fps_conditioning))  # [B, nt*dim]
        v_fps_proj = self.v_fps_up(self.v_fps_down(fps_conditioning))  # [B, nt*dim]

        # 2) normalize K' in full dim (match base K scale)
        B = q.size(0)
        k_fps_proj = k_fps_proj.view(B * self.num_tokens, self.dim)
        k_fps_proj = self.norm_k_fps(k_fps_proj)                       # RMSNorm (trainable scale)
        k_fps_proj = k_fps_proj.view(B, self.num_tokens * self.dim)

        # 3) apply LoRA scale AFTER norm so it actually changes amplitude
        k_fps_proj = k_fps_proj * self.lora_scale
        v_fps_proj = v_fps_proj * self.lora_scale

        # 4) reshape to attention layout
        k_fps = k_fps_proj.view(B, self.num_tokens, self.num_heads, self.head_dim)
        v_fps = v_fps_proj.view(B, self.num_tokens, self.num_heads, self.head_dim)
        
        # ALWAYS compute FPS attention (deterministic operation)
        y_fps = flash_attention(q, k_fps, v_fps, k_lens=None)
        
        # Debug: Print y_fps/y_text magnitude ratio every 1000 forward passes (less spam)
        # Debug: Track FPS vs text ratio (detached from gradient computation)
        if hasattr(self, 'debug_counter'):
            self.debug_counter += 1
            if self.debug_counter % 1000 == 0:  # Every 1000 forward passes
                with torch.no_grad():  # Completely detached from gradient computation
                    # Create detached copies to avoid gradient issues
                    y_text_detached = y_text.detach()
                    y_fps_detached = y_fps.detach()
                    
                    y_text_norm = torch.norm(y_text_detached).item()
                    y_fps_norm = torch.norm(y_fps_detached).item()
                    ratio = y_fps_norm / (y_text_norm + 1e-8)  # Avoid division by zero
                    
                    # Get gate value based on gate mode
                    if self.gate_mode in ['sigmoid', 'identity', 'relu', 'silu', 'softplus']:
                        gate_alpha_detached = self.gate_alpha.detach()
                        if self.gate_mode == 'sigmoid':
                            gate_val = torch.sigmoid(gate_alpha_detached).item()
                        elif self.gate_mode == 'identity':
                            gate_val = gate_alpha_detached.item()
                        elif self.gate_mode == 'relu':
                            gate_val = torch.relu(gate_alpha_detached).item()
                        elif self.gate_mode == 'silu':
                            gate_val = torch.nn.functional.silu(gate_alpha_detached).item()
                        elif self.gate_mode == 'softplus':
                            gate_val = torch.nn.functional.softplus(gate_alpha_detached).item()
                    elif self.gate_mode == 'fixed':
                        gate_val = self.gate_fixed.item()
                    else:
                        gate_val = 0.0  # fallback
                    
                    print(f"[FPS_RATIO] Forward {self.debug_counter}: ||y_fps||/||y_text|| = {ratio:.6f}, gate = {gate_val:.4f} ({self.gate_mode})")
        
        # Apply FPS scaling if enabled (match y_fps magnitude to y_text)
        if self.fps_scale:
            # Compute magnitude of y_text and y_fps
            y_text_norm = torch.norm(y_text)
            y_fps_norm = torch.norm(y_fps)

            # Avoid division by zero: add small constant if y_fps_norm is very small
            # This prevents instability when y_fps is near zero at initialization
            y_fps_norm_safe = y_fps_norm + 1e-8

            # Scale y_fps to match y_text magnitude
            scale_factor = y_text_norm / y_fps_norm_safe
            y_fps = y_fps * scale_factor

        # Apply activation function based on gate mode
        if self.gate_mode == 'sigmoid':
            gate = torch.sigmoid(self.gate_alpha)
        elif self.gate_mode == 'identity':
            gate = self.gate_alpha
        elif self.gate_mode == 'relu':
            gate = torch.relu(self.gate_alpha)
        elif self.gate_mode == 'silu':
            gate = torch.nn.functional.silu(self.gate_alpha)
        elif self.gate_mode == 'softplus':
            gate = torch.nn.functional.softplus(self.gate_alpha)
        elif self.gate_mode == 'fixed':
            gate = self.gate_fixed
        else:
            raise ValueError(f"Unknown gate_mode: {self.gate_mode}. Must be one of: 'sigmoid', 'identity', 'relu', 'silu', 'softplus', 'fixed'")

        # Apply warmup schedule: w(t) = min(1, t/K)
        # During training, gradually ramp up FPS influence from 0 to full strength
        if self.training and self.warmup_steps > 0:
            warmup_factor = torch.clamp(self.current_step.float() / self.warmup_steps, max=1.0)
            # Increment step counter (only during training)
            self.current_step += 1
        else:
            # During inference or if warmup disabled, use full strength
            warmup_factor = 1.0

        # Compute final output: y = y_text + (w(t) * g(alpha)) * y_fps_scaled
        y_combined = y_text + (warmup_factor * gate) * y_fps

        return y_combined


WAN_CROSSATTENTION_CLASSES = {
    # T2V and all Wan2.2 cross attn
    'default': WanCrossAttention,
    # Wan2.1 I2V only
    'wan2_1_i2v_cross_attn': WanI2VCrossAttention,
}


class WanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 fps_adapter_config=None):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.fps_adapter_config = fps_adapter_config

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm,
                                          eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                      num_heads,
                                                                      (-1, -1),
                                                                      qk_norm,
                                                                      eps)
        
        # FPS adapter (only for selected blocks)
        if fps_adapter_config is not None:
            self.fps_adapter = FPSCrossAttentionAdapter(
                dim=dim,
                num_heads=num_heads,
                fps_conditioning_dim=fps_adapter_config['fps_conditioning_dim'],
                rank=fps_adapter_config['rank'],
                gate_init=fps_adapter_config['gate_init'],
                num_tokens=fps_adapter_config.get('num_tokens', 1),  # Default to 1 for backward compatibility
                lora_alpha=fps_adapter_config.get('lora_alpha', 16),  # Default to 16 for LoRA scaling
                gate_mode=fps_adapter_config.get('gate_mode', 'fixed'),  # Default to fixed mode
                gate_fixed_value=fps_adapter_config.get('gate_fixed_value', 0.5),  # Default fixed value
                fps_scale=fps_adapter_config.get('fps_scale', False),  # Default to False
                warmup_steps=fps_adapter_config.get('warmup_steps', 100)  # Default to 100 steps
            )
        else:
            self.fps_adapter = None
            
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        fps_conditioning=None,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, L1, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)

        # self-attention
        y = self.self_attn(
            self.norm1(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2),
            seq_lens, grid_sizes, freqs)
        x = x + y * e[2].squeeze(2)

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e, fps_conditioning):
            if self.fps_adapter is not None:
                # Use FPS-enhanced cross-attention (fps_conditioning is ALWAYS valid)
                normed_x = self.norm3(x)
                b, n, d = normed_x.size(0), self.num_heads, self.dim // self.num_heads
                
                # Compute Q, K_text, V_text for standard cross-attention
                q = self.cross_attn.norm_q(self.cross_attn.q(normed_x)).view(b, -1, n, d)
                k_text = self.cross_attn.norm_k(self.cross_attn.k(context)).view(b, -1, n, d)
                v_text = self.cross_attn.v(context).view(b, -1, n, d)
                
                # Apply FPS adapter for combined attention
                attn_out = self.fps_adapter(q, k_text, v_text, fps_conditioning, context_lens)
                
                # Apply output projection
                attn_out = attn_out.flatten(2)
                attn_out = self.cross_attn.o(attn_out)
                x = x + attn_out
            else:
                # Standard cross-attention without FPS conditioning
                x = x + self.cross_attn(self.norm3(x), context, context_lens)
                
            y = self.ffn(self.norm2(x) * (1 + e[4].squeeze(2)) + e[3].squeeze(2))
            x = x + y * e[5].squeeze(2)
            return x

        x = cross_attn_ffn(x, context, context_lens, e, fps_conditioning)
        return x


class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, L1, C]
        """
        with torch.amp.autocast('cuda', dtype=torch.float32):
            e = (self.modulation.unsqueeze(0) + e.unsqueeze(2)).chunk(2, dim=2)
            x = (
                self.head(
                    self.norm(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2)))
        return x


class MLPProj(torch.nn.Module):

    def __init__(self, in_dim, out_dim, flf_pos_emb=False):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim), torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(), torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim))
        if flf_pos_emb:  # NOTE: we only use this for `flf2v`
            self.emb_pos = nn.Parameter(
                torch.zeros(1, FIRST_LAST_FRAME_CONTEXT_TOKEN_NUMBER, 1280))

    def forward(self, image_embeds):
        if hasattr(self, 'emb_pos'):
            bs, n, d = image_embeds.shape
            image_embeds = image_embeds.view(-1, 2 * n, d)
            image_embeds = image_embeds + self.emb_pos
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class WanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    ]
    _no_split_modules = ['WanAttentionBlock']

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6,
                 fps_adapter_rank=8,
                 fps_adapter_gate_init=0.0,
                 fps_condition_blocks="deepest_third",
                 fps_adapter_num_tokens=1,
                 fps_tau_transform="log1p",
                 fps_tau_scale=0.33333334,
                 fps_reference_fps=240.0,
                 fps_embed_dim=256,
                 fps_condition_hidden=64,
                 fps_lora_alpha=16,
                 fps_gate_mode='fixed',
                 fps_gate_fixed_value=0.5,
                 fps_scale=False,
                 fps_warmup_steps=100):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video) or 'flf2v' (first-last-frame-to-video) or 'vace'
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v', 'flf2v', 'vace', 'i2v_v2', 'ti2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        
        # FPS conditioning configuration
        self.fps_tau_transform = fps_tau_transform
        self.fps_tau_scale = fps_tau_scale
        self.fps_reference_fps = fps_reference_fps

        # Determine number of conditions from fps_tau_transform
        # If it's a list, its length is the number of conditions
        # Otherwise, it's a single condition (default behavior)
        if isinstance(fps_tau_transform, list):
            num_conditions = len(fps_tau_transform)
        else:
            num_conditions = 1

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # FPS conditioning encoder with special initialization
        # num_conditions parameter allows multi-condition support
        self.fps_conditioning = FpsConditioning(
            out_dim=fps_embed_dim,
            hidden=fps_condition_hidden,
            num_conditions=num_conditions
        )

        # blocks
        if model_type in ('i2v', 'flf2v'):
            cross_attn_type = 'wan2_1_i2v_cross_attn'
        else:
            cross_attn_type = 'default'
            
        # determine which blocks get fps adapter (deepest third or all)
        if isinstance(fps_condition_blocks, str) and fps_condition_blocks == "deepest_third":
            fps_blocks_count = num_layers // 3
            fps_block_indices = set(range(num_layers - fps_blocks_count, num_layers))
        elif isinstance(fps_condition_blocks, str) and fps_condition_blocks == "all":
            fps_block_indices = set(range(num_layers))
        elif isinstance(fps_condition_blocks, int):
            fps_blocks_count = min(fps_condition_blocks, num_layers)
            fps_block_indices = set(range(num_layers - fps_blocks_count, num_layers))
        else:
            fps_block_indices = set()
            
        # fps conditioning MLP has configurable output dim (fps_embed_dim)
        fps_conditioning_dim = fps_embed_dim
        
        self.blocks = nn.ModuleList([
            WanAttentionBlock(
                cross_attn_type, dim, ffn_dim, num_heads,
                window_size, qk_norm, cross_attn_norm, eps,
                fps_adapter_config={
                    'fps_conditioning_dim': fps_conditioning_dim,
                    'rank': fps_adapter_rank,
                    'gate_init': fps_adapter_gate_init,
                    'num_tokens': fps_adapter_num_tokens,
                    'lora_alpha': fps_lora_alpha,
                    'gate_mode': fps_gate_mode,
                    'gate_fixed_value': fps_gate_fixed_value,
                    'fps_scale': fps_scale,
                    'warmup_steps': fps_warmup_steps
                } if i in fps_block_indices else None
            )
            for i in range(num_layers)
        ])
        
        # Count FPS adapter parameters silently
        total_fps_adapter_params = 0
        for i, block in enumerate(self.blocks):
            if hasattr(block, 'fps_adapter') and block.fps_adapter is not None:
                for name, param in block.fps_adapter.named_parameters():
                    total_fps_adapter_params += param.numel()
        
        # Verify FPS adapter initialization silently
        verified_adapters = 0
        for i, block in enumerate(self.blocks):
            if hasattr(block, 'fps_adapter') and block.fps_adapter is not None:
                is_correct = block.fps_adapter.verify_initialization()
                if is_correct:
                    verified_adapters += 1

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
                               dim=1)

        if model_type == 'i2v' or model_type == 'flf2v':
            self.img_emb = MLPProj(1280, dim, flf_pos_emb=model_type == 'flf2v')

        # initialize weights
        self.init_weights()

    # Removed forward() because we don't use it due to pipeline parallelism.

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init - skip FpsConditioning modules and their children (they handle their own initialization)
        for name, m in self.named_modules():
            if isinstance(m, FpsConditioning):
                continue
            # Skip Linear modules that are children of FpsConditioning
            if isinstance(m, nn.Linear) and 'fps_conditioning' not in name:
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)