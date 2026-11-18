import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Adaptive Layer Normalization (adaLN) modulation function.
    
    Applies element-wise affine transformation conditioned on diffusion timesteps:
        modulate(x, shift, scale) = x * (1 + scale) + shift
    
    Args:
        x: (B, N, D) - Normalized token features.
        shift: (B, D) - Shift vector predicted by adaLN MLP from timestep embedding.
        scale: (B, D) - Scale vector predicted by adaLN MLP from timestep embedding.
    Returns:
        (B, N, D) - Modulated token representations.
    """
    # Unsqueeze dim 1 so (B, D) broadcasts across all N spatial tokens (B, 1, D)
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int | Tuple[int, int]) -> np.ndarray:
    """
    Generate 2D sinusoidal position embeddings for a 2D spatial grid (as in DiT and MAE).
    
    Decomposes the 2D spatial grid into two independent 1D coordinate grids:
      - Half the channels (embed_dim // 2) encode vertical height (H) coordinates.
      - Half the channels (embed_dim // 2) encode horizontal width (W) coordinates.
    
    Args:
        embed_dim: Total embedding dimension (e.g., 768). Must be an even number.
        grid_size: Int or (H_grid, W_grid) tuple representing spatial token resolution (e.g., 32x32).
    Returns:
        pos_embed: (grid_h * grid_w, embed_dim) numpy array of continuous sinusoidal coordinates.
    """
    if isinstance(grid_size, int):
        grid_h = grid_w = grid_size
    else:
        grid_h, grid_w = grid_size

    # Create coordinate vectors along height and width
    grid_h_coords = np.arange(grid_h, dtype=np.float32)
    grid_w_coords = np.arange(grid_w, dtype=np.float32)
    
    # Meshgrid creates a 2D coordinate matrix for all (x, y) spatial locations
    grid = np.meshgrid(grid_w_coords, grid_h_coords)  # w, h
    grid = np.stack(grid, axis=0)                     # Shape: (2, grid_h, grid_w)
    grid = grid.reshape([2, 1, grid_h, grid_w])

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    """Encodes grid[0] (width) and grid[1] (height) into concatenated 1D sinusoidal embeddings."""
    assert embed_dim % 2 == 0, f"embed_dim must be divisible by 2, got {embed_dim}"
    # Vertical coordinates encoded in first half of channels
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
    # Horizontal coordinates encoded in second half of channels
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    # Concatenate to form complete 2D positional representation
    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    """
    Computes standard sinusoidal position embeddings (Vaswani et al., 2017) along 1 dimension:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # Geometric progression of frequencies

    pos = pos.reshape(-1)  # Flatten 2D grid to 1D vector of coordinates
    out = np.einsum("m,d->md", pos, omega)  # Outer product between position and frequencies

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


class TimestepEmbedder(nn.Module):
    """
    Diffusion Timestep Embedder:
    Projects continuous diffusion timesteps t in [0, 1000] into the DiT hidden dimension (D=768).
    
    Structure:
      1. Sinusoidal Embedding: Maps scalar t into high-dimensional geometric frequencies (d=256).
      2. Multi-Layer Perceptron: 2-layer MLP with SiLU activation mapping frequencies to hidden_size.
    """
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    @staticmethod
    def sinusoidal_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        """Calculates continuous sinusoidal positional frequencies for scalar timesteps."""
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2 == 1:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # 1. Compute sinusoidal frequencies for each timestep in batch
        t_freq = self.sinusoidal_embedding(t, self.frequency_embedding_size).to(dtype=self.mlp[0].weight.dtype)
        # 2. Pass through non-linear projection to generate conditioning vector c_t: (B, hidden_size)
        return self.mlp(t_freq)


class PreConvStem(nn.Module):
    """
    Residual Pre-Convolutional Stem:
    Applied directly to the 4-channel VAE latents prior to patch projection.
    
    Why this is used:
      - Standard DiT models apply a strided patch projection that ignores continuous pixel-to-pixel
        gradients across patch boundaries, leading to grid artifacts.
      - This lightweight 2-layer 3x3 conv module learns continuous edge and smoothing filters.
      - Zero-initialized on the second convolution so that at epoch 0 it behaves as an exact
        mathematical identity (x + 0 = x), preserving pre-trained ViT weight transfer completely.
    """
    def __init__(self, in_channels: int = 4, hidden_channels: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.act = nn.SiLU()
        self.conv2 = nn.Conv2d(hidden_channels, in_channels, kernel_size=3, padding=1)

        # Zero-initialize conv2 weights and biases to guarantee identity mapping at step 0
        nn.init.constant_(self.conv2.weight, 0)
        nn.init.constant_(self.conv2.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual connection: x + Conv_residual(x)
        return x + self.conv2(self.act(self.conv1(x)))


class PostConvSmoothing(nn.Module):
    """
    Residual Post-Convolutional Smoothing Head:
    Applied after unpatchification to the predicted noise tensor epsilon_theta.
    
    Why this is used:
      - Tiling 2x2 patches during unpatchify can leave high-frequency checkerboard phase seams.
      - This 2-layer residual 3x3 conv head operates in spatial latent space to smooth boundary seams.
      - Also zero-initialized so the initial model predicts clean, unperturbed noise residuals.
    """
    def __init__(self, out_channels: int = 4, hidden_channels: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(out_channels, hidden_channels, kernel_size=3, padding=1)
        self.act = nn.SiLU()
        self.conv2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1)

        # Zero-initialize to ensure exact identity behavior initially
        nn.init.constant_(self.conv2.weight, 0)
        nn.init.constant_(self.conv2.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual connection: x + Conv_residual(x)
        return x + self.conv2(self.act(self.conv1(x)))


class PatchEmbed(nn.Module):
    """
    2D Image Latent to Patch Embedding:
    Splits continuous 4-channel spatial latents into a sequence of flattened token vectors.
    
    Configuration Details:
      - Symmetric Non-Overlapping Mode (overlapping=False, kernel=2, stride=2, padding=0):
        Maps each 2x2 latent tile directly to 1 token vector. This mirrors FinalLayer.unpatchify
        in a 1:1 isomorphic grid layout, eliminating phase offsets that cause wavy standing ripples.
      - Overlapping Mode (overlapping=True, kernel=3, stride=2, padding=1):
        Uses a 3x3 kernel with stride 2 and padding 1 to share 1-pixel border context across tokens.
    """
    def __init__(
        self,
        sample_size: int = 48,
        patch_size: int = 2,
        in_channels: int = 4,
        hidden_size: int = 768,
        overlapping: bool = True,
    ):
        super().__init__()
        self.sample_size = sample_size
        self.patch_size = patch_size
        self.grid_size = sample_size // patch_size
        self.num_patches = self.grid_size * self.grid_size

        if overlapping and patch_size == 2:
            # Overlapping 3x3 kernel with stride 2
            self.proj = nn.Conv2d(in_channels, hidden_size, kernel_size=3, stride=patch_size, padding=1)
        else:
            # Symmetric non-overlapping 2x2 patch projection matching unpatchify
            self.proj = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape:  (B, C, H, W) e.g., (B, 4, 64, 64)
        # Conv projection: (B, hidden_size, H/p, W/p) e.g., (B, 768, 32, 32)
        x = self.proj(x)
        # Flatten spatial dimensions: (B, hidden_size, num_patches) -> Transpose to (B, num_patches, hidden_size)
        x = x.flatten(2).transpose(1, 2)
        return x


class DiTAttention(nn.Module):
    """
    Multi-Head Attention module for DiT blocks:
    Supports Spatial Self-Attention (within image tokens) and Cross-Attention (to Perceiver context)
    using PyTorch native Scaled Dot Product Attention (SDPA).
    """
    def __init__(
        self,
        dim: int,
        context_dim: Optional[int] = None,
        num_heads: int = 12,
        qkv_bias: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        context_dim = context_dim if context_dim is not None else dim

        # qkv_bias=True matches the pretrained Vision Transformer (ViT-B) weights
        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_k = nn.Linear(context_dim, dim, bias=qkv_bias)
        self.to_v = nn.Linear(context_dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.proj_drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N_q, _ = x.shape
        is_cross = context is not None
        context = context if is_cross else x
        N_kv = context.shape[1]

        # 1. Project inputs to Q, K, V and reshape to (B, heads, seq_len, head_dim)
        q = self.to_q(x).view(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.to_k(context).view(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.to_v(context).view(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)

        # 2. Handle attention mask: invert boolean mask because SDPA expects True for valid positions
        attn_mask = None
        if mask is not None:
            if mask.dtype == torch.bool:
                valid_mask = ~mask
                if valid_mask.ndim == 2:
                    attn_mask = valid_mask.unsqueeze(1).unsqueeze(2)
                elif valid_mask.ndim == 3:
                    attn_mask = valid_mask.unsqueeze(1)
                else:
                    attn_mask = valid_mask
            else:
                attn_mask = mask.unsqueeze(1).unsqueeze(2) if mask.ndim == 2 else mask

        # 3. High-performance attention via PyTorch SDPA kernel
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
        )

        # 4. Concatenate multi-head outputs and project back to hidden_size
        out = out.transpose(1, 2).contiguous().view(B, N_q, -1)
        return self.proj_drop(self.proj(out))


class DiTBlock(nn.Module):
    """
    Diffusion Transformer (DiT) Block:
    The core computational unit of the DiT backbone.
    
    Innovations in this block:
      1. Adaptive LayerNorm (adaLN-Zero): Timestep conditioning vector c modulates features
         via 7 predicted parameters: shift/scale/gate for MSA, shift/scale/gate for MLP, and gate for Cross-Attn.
      2. Gated Cross-Attention: Dedicated cross-attention mechanism attending directly to the
         compact Perceiver IO multimodal story tokens.
      3. Deep Positional Re-Anchoring: Re-injects 2D sinusoidal positional coordinates into the
         residual stream before self-attention with a guaranteed non-cancellable floor (0.10 + ReLU(pos_gate)).
         This solves the coordinate dilution problem across deep layers without zeroing out under BF16 precision.
    """
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        context_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        context_dim = context_dim if context_dim is not None else hidden_size

        # Pre-LayerNorm for Self-Attention (elementwise_affine=False because adaLN predicts scale & shift)
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.self_attn = DiTAttention(dim=hidden_size, context_dim=None, num_heads=num_heads, dropout=dropout)

        # Pre-LayerNorm and Cross-Attention to Perceiver IO fused story context
        self.norm_cross = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.cross_attn = DiTAttention(dim=hidden_size, context_dim=context_dim, num_heads=num_heads, dropout=dropout)

        # Pre-LayerNorm for MLP
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, hidden_size),
            nn.Dropout(dropout),
        )

        # adaLN-Zero modulation MLP: Predicts 7 conditioning vectors of size (hidden_size):
        # (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp, gate_cross)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 7 * hidden_size, bias=True)
        )

        # Zero-initialize the modulation MLP and residual projections (adaLN-Zero philosophy):
        # When gates are zero, the entire block acts as an exact identity mapping at initialization,
        # allowing deep models to begin training with total stability.
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.self_attn.proj.weight, 0)
        nn.init.constant_(self.self_attn.proj.bias, 0)
        nn.init.constant_(self.cross_attn.proj.weight, 0)
        nn.init.constant_(self.cross_attn.proj.bias, 0)
        nn.init.constant_(self.mlp[-2].weight, 0)
        nn.init.constant_(self.mlp[-2].bias, 0)

        # Learnable gate for Deep Positional Re-Anchoring
        self.pos_gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        pos_embed: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, hidden_size) - Spatial patch tokens.
            c: (B, hidden_size) - Timestep conditioning vector from TimestepEmbedder.
            context: (B, M, context_dim) - Perceiver IO fused story context tokens.
            pos_embed: (1, N, hidden_size) - 2D sinusoidal position embedding for deep re-anchoring.
        """
        # ── 1. Deep Positional Re-Anchoring ──
        # In deep transformers, spatial coordinate information dilutes across layers, leading to
        # spatial drift / wavy building walls. We re-inject coordinates before each self-attention block.
        # We enforce a guaranteed non-cancellable base anchor (0.10) + non-negative learned residual (ReLU(pos_gate)).
        # This prevents gradient descent from canceling the coordinate signal under BF16 precision.
        if pos_embed is not None:
            x = x + (0.10 + F.relu(self.pos_gate)) * pos_embed.to(device=x.device, dtype=x.dtype)

        # ── 2. Compute adaLN Modulation Vectors ──
        # Predict scale, shift, and gate multipliers from timestep conditioning c
        mod = self.adaLN_modulation(c)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp, gate_cross = mod.chunk(7, dim=1)

        # ── 3. Spatial Self-Attention (with adaLN scale/shift and output gating) ──
        # mod_norm1 = modulate(norm1(x), shift_msa, scale_msa)
        # x = x + gate_msa * Attention(mod_norm1)
        x = x + gate_msa.unsqueeze(1) * self.self_attn(modulate(self.norm1(x), shift_msa, scale_msa))

        # ── 4. Cross-Attention to Multimodal Story Context ──
        if context is not None:
            # Gated cross-attention allows the network to smoothly blend story conditioning
            x = x + (1.0 + gate_cross.unsqueeze(1)) * self.cross_attn(self.norm_cross(x), context=context)

        # ── 5. Modulated Feed-Forward MLP ──
        # mod_norm2 = modulate(norm2(x), shift_mlp, scale_mlp)
        # x = x + gate_mlp * MLP(mod_norm2)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    Final Layer of DiT:
    Transforms output transformer tokens back into the continuous spatial latent noise prediction.
    
    Structure:
      1. adaLN Modulation: Scales and shifts final normalized tokens based on diffusion timestep.
      2. Linear Projection: Maps hidden_size (768) -> (patch_size * patch_size * out_channels) = (2*2*4 = 16).
      3. Unpatchify: Reshapes flattened patch tokens back into a 4-channel spatial latent grid (B, 4, H, W).
    """
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        self.patch_size = patch_size
        self.out_channels = out_channels

        # Zero-initialize the final projection so the initial model predicts pure zero residual
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)

    def unpatchify(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """
        Reconstructs 2D spatial latents from a 1D sequence of patch tokens.
        
        Args:
            x: (B, N, patch_size * patch_size * out_channels) - Predicted patch values.
            h, w: Height and width of the target spatial latent (e.g., 64x64).
        Returns:
            imgs: (B, out_channels, h, w) - Continuous spatial noise prediction tensor.
        """
        c = self.out_channels
        p = self.patch_size
        gh, gw = h // p, w // p  # Grid dimensions (e.g., 32x32 for 64x64 with patch_size=2)

        # Reshape to separate spatial patch grid indices from intra-patch pixel coordinates:
        # (B, N, p*p*c) -> (B, gh, gw, p, p, c)
        x = x.reshape(shape=(x.shape[0], gh, gw, p, p, c))
        # Rearrange dimensions using Einstein summation:
        # n: batch, h: grid_h, w: grid_w, p: patch_h, q: patch_w, c: channel
        # Permute to (n, c, h, p, w, q)
        x = torch.einsum("nhwpqc->nchpwq", x)
        # Flatten (h, p) into full Height (gh*p) and (w, q) into full Width (gw*p):
        imgs = x.reshape(shape=(x.shape[0], c, gh * p, gw * p))
        return imgs

    def forward(self, x: torch.Tensor, c: torch.Tensor, h: int, w: int) -> torch.Tensor:
        # 1. Modulate normalized tokens with timestep scale and shift
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        # 2. Linear projection to intra-patch pixel values
        x = self.linear(x)
        # 3. Unpatchify into 2D spatial latent map
        return self.unpatchify(x, h, w)
