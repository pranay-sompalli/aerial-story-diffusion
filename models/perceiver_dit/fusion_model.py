import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from .dit import (
    DiTBlock,
    FinalLayer,
    PatchEmbed,
    PostConvSmoothing,
    PreConvStem,
    TimestepEmbedder,
    get_2d_sincos_pos_embed,
)
from .perceiver_io import PerceiverIO


@dataclass
class PerceiverIODiTOutput:
    """
    Output container dataclass matching HuggingFace diffusers UNet2DConditionOutput.
    Ensures that calling `model(sample, timestep, ...).sample` works seamlessly as a
    drop-in replacement for Stable Diffusion's standard UNet.
    """
    sample: torch.FloatTensor


class PerceiverIODiT(nn.Module):
    """
    PerceiverIO-DiT Fusion Model:
    A modular, high-efficiency Diffusion Transformer (DiT-B) fused with a Perceiver IO
    cross-attention condition resampler, designed specifically for multi-frame aerial story continuation.
    
    Architectural Pipeline:
      1. Pre-Conv Stem: 3x3 residual convolution preserving continuous road and vehicle textures.
      2. Patch Projection: 2x2 symmetric patchification mapping (B, 4, 64, 64) -> (B, 1024, 768).
      3. 2D Sinusoidal Position Embeddings: Analytical continuous spatial coordinate grid.
      4. Diffusion Timestep Embedder: Sinusoidal MLP embedding diffusion steps t into (B, 768).
      5. Perceiver IO Resampler: Distills variable-length story context (CLIP text + BLIP frame history)
         into exactly M=64 distilled cross-modal conditioning tokens.
      6. DiT Transformer Backbone: 12 stacked DiT blocks with adaLN-Zero timestep modulation,
         spatial multi-head self-attention, and gated cross-attention to the Perceiver tokens.
      7. Deep Positional Re-Anchoring: Enforces non-cancellable coordinate injection (0.10 + ReLU(gate))
         at every layer to prevent coordinate decay without zeroing out under BF16 precision.
      8. Final Layer & Unpatchify: Reconstructs the 4-channel spatial noise prediction.
      9. Post-Conv Smoothing: Cancels high-frequency patch boundary seams.
    """
    def __init__(
        self,
        sample_size: int = 48,          # Spatial latent resolution (e.g., 64 for 512x512 RGB images)
        patch_size: int = 2,           # Intra-latent patch size (p=2 -> 2x2 latent cells = 16x16 RGB pixels)
        in_channels: int = 4,          # Input VAE latent channels
        out_channels: int = 4,         # Output predicted noise channels
        hidden_size: int = 768,        # DiT hidden dimension (DiT-B: 768, DiT-L: 1024)
        depth: int = 12,               # Number of DiT transformer blocks
        num_heads: int = 12,           # Attention heads (head_dim = 768 // 12 = 64)
        mlp_ratio: float = 4.0,        # FFN expansion ratio (768 * 4 = 3072)
        perceiver_num_latents: int = 64,# Bottleneck latent queries in Perceiver IO (M=64)
        perceiver_latent_dim: int = 768,# Latent dimension inside Perceiver IO
        perceiver_depth: int = 2,      # Latent self-attention layers in Perceiver IO
        perceiver_num_heads: int = 8,  # Attention heads in Perceiver IO
        context_dim: int = 768,        # Dimension of input multimodal features (CLIP + BLIP)
        dropout: float = 0.0,
        use_conv_stem: bool = True,    # Enable residual conv stem and smoothing head
        overlapping_patch: bool = False,# Symmetric non-overlapping 2x2 patch projection
    ):
        super().__init__()
        self.sample_size = sample_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.gradient_checkpointing = False

        # ── 1. Residual Convolutional Stem & Smoothing Head ──
        # Preserves continuous pixel-to-pixel boundaries across patch cuts in aerial imagery
        self.use_conv_stem = use_conv_stem
        if use_conv_stem:
            self.pre_conv = PreConvStem(in_channels=in_channels, hidden_channels=32)
            self.post_conv = PostConvSmoothing(out_channels=out_channels, hidden_channels=32)
        else:
            self.pre_conv = nn.Identity()
            self.post_conv = nn.Identity()

        # ── 2. Patch Projection ──
        # Symmetric 2x2 non-overlapping projection matching FinalLayer.unpatchify 1:1
        self.patch_embed = PatchEmbed(
            sample_size=sample_size,
            patch_size=patch_size,
            in_channels=in_channels,
            hidden_size=hidden_size,
            overlapping=overlapping_patch,
        )

        # ── 3. 2D Sinusoidal Spatial Positional Embeddings ──
        # Registered as non-persistent buffer so checkpoints remain resolution-agnostic
        grid_size = sample_size // patch_size
        pos_embed = get_2d_sincos_pos_embed(hidden_size, grid_size)
        self.register_buffer("pos_embed", torch.from_numpy(pos_embed).float().unsqueeze(0), persistent=False)

        # ── 4. Diffusion Timestep Embedder ──
        # Maps diffusion timesteps t in [0, 1000] into continuous vectors: (B, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)

        # ── 5. Perceiver IO Multimodal Condition Resampler ──
        # Distills variable-length story tokens into exactly M=64 conditioning tokens
        self.perceiver = PerceiverIO(
            num_latents=perceiver_num_latents,
            latent_dim=perceiver_latent_dim,
            context_dim=context_dim,
            out_dim=hidden_size,
            depth=perceiver_depth,
            heads=perceiver_num_heads,
            dropout=dropout,
        )

        # ── 6. DiT Transformer Backbone Blocks ──
        # 12 stacked blocks with adaLN modulation, self-attention, and cross-attention
        self.blocks = nn.ModuleList([
            DiTBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                context_dim=hidden_size,
                dropout=dropout,
            )
            for _ in range(depth)
        ])

        # ── 7. Final Layer and Unpatchification Head ──
        # Projects tokens back to (B, out_channels, H, W)
        self.final_layer = FinalLayer(
            hidden_size=hidden_size,
            patch_size=patch_size,
            out_channels=out_channels,
        )

    def enable_gradient_checkpointing(self):
        """Enables activation checkpointing across DiT blocks to conserve VRAM during training."""
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        """Disables activation checkpointing."""
        self.gradient_checkpointing = False

    def load_pretrained_vit_b(self, vit_model_name: str = "vit_base_patch16_224") -> None:
        """
        Transfer pre-trained Vision Transformer (ViT-B/16 ImageNet-21k) weights into DiT-B blocks.

        Why this accelerates training:
          - ViT-B and DiT-B share identical transformer block geometry (12 layers, hidden=768, heads=12, MLP=3072).
          - Transferring ImageNet visual representations into Self-Attention (Q, K, V, Proj) and MLP (fc1, fc2)
            gives the model powerful spatial filtering and object representation priors from day 1.

        Transferred parameters (per block):
          - `blocks.{i}.self_attn.to_q/k/v`  <-- ViT `blocks.{i}.attn.qkv` (split 3-way)
          - `blocks.{i}.self_attn.proj`       <-- ViT `blocks.{i}.attn.proj`
          - `blocks.{i}.mlp.{0,3}`           <-- ViT `blocks.{i}.mlp.fc1` and `fc2`

        Intentionally NOT transferred (diffusion/story-specific, remain zero-initialized):
          - `t_embedder`, `adaLN_modulation`: Timestep conditioning (adaLN-Zero)
          - `cross_attn`: Cross-attention to Perceiver IO multimodal context
          - `perceiver`: Multimodal story resampler
          - `final_layer`: Final noise prediction head
          - `patch_embed`: 4-channel latent input vs ViT's 3-channel RGB
        """
        try:
            import timm
        except ImportError:
            raise ImportError(
                "timm is required for ViT-B weight transfer. Install with: pip install timm"
            )

        print(f">>> Loading pre-trained ViT-B weights from timm: '{vit_model_name}'...", flush=True)
        vit = timm.create_model(vit_model_name, pretrained=True)
        vit_state = vit.state_dict()

        transferred = 0
        for i, block in enumerate(self.blocks):
            prefix = f"blocks.{i}"

            # ── Self-Attention: ViT stores fused qkv weights -> split into to_q, to_k, to_v ──
            qkv_weight_key = f"{prefix}.attn.qkv.weight"
            qkv_bias_key   = f"{prefix}.attn.qkv.bias"
            proj_weight_key = f"{prefix}.attn.proj.weight"
            proj_bias_key   = f"{prefix}.attn.proj.bias"

            if qkv_weight_key in vit_state:
                # ViT qkv tensor has shape (3 * hidden_size, hidden_size). Split evenly into Q, K, V.
                qkv_w = vit_state[qkv_weight_key]   # Shape: (3 * 768, 768)
                qkv_b = vit_state[qkv_bias_key]     # Shape: (3 * 768,)
                D = qkv_w.shape[0] // 3

                with torch.no_grad():
                    # Copy Q, K, V weight slices
                    block.self_attn.to_q.weight.copy_(qkv_w[:D])
                    block.self_attn.to_k.weight.copy_(qkv_w[D:2*D])
                    block.self_attn.to_v.weight.copy_(qkv_w[2*D:])
                    # Copy Q, K, V bias slices if present
                    if block.self_attn.to_q.bias is not None:
                        block.self_attn.to_q.bias.copy_(qkv_b[:D])
                        block.self_attn.to_k.bias.copy_(qkv_b[D:2*D])
                        block.self_attn.to_v.bias.copy_(qkv_b[2*D:])
                transferred += 3

            # ── Self-Attention: Output projection matrix ──
            if proj_weight_key in vit_state:
                with torch.no_grad():
                    block.self_attn.proj.weight.copy_(vit_state[proj_weight_key])
                    if block.self_attn.proj.bias is not None:
                        block.self_attn.proj.bias.copy_(vit_state[proj_bias_key])
                transferred += 1

            # ── Feed-Forward MLP: fc1 -> mlp[0], fc2 -> mlp[3] ──
            for fc_key, layer_idx in [(f"{prefix}.mlp.fc1", 0), (f"{prefix}.mlp.fc2", 3)]:
                w_key = f"{fc_key}.weight"
                b_key = f"{fc_key}.bias"
                if w_key in vit_state:
                    with torch.no_grad():
                        block.mlp[layer_idx].weight.copy_(vit_state[w_key])
                        block.mlp[layer_idx].bias.copy_(vit_state[b_key])
                    transferred += 1

        del vit  # Free ViT model memory
        print(
            f">>> ViT-B weight transfer complete: {transferred} weight tensors transferred "
            f"across {len(self.blocks)} DiT blocks.\n"
            f"    Diffusion-specific layers (adaLN, cross_attn, perceiver, t_embedder, "
            f"final_layer, patch_embed) remain zero-initialized.",
            flush=True
        )

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[PerceiverIODiTOutput, Tuple[torch.Tensor]]:
        """
        Forward pass matching the HuggingFace UNet2DConditionModel signature.

        Args:
            sample: (B, in_channels, H, W) - Noisy latent feature maps (e.g., B x 4 x 64 x 64).
            timestep: (B,) - Diffusion timesteps.
            encoder_hidden_states: (B, seq_len, context_dim) - Concatenated CLIP text + BLIP frame history tokens.
            encoder_attention_mask: (B, seq_len) - Boolean mask (True indicates padding/masked-out positions).
            return_dict: Whether to return PerceiverIODiTOutput dataclass or raw tuple.
        Returns:
            PerceiverIODiTOutput with `.sample` holding the predicted noise tensor (B, out_channels, H, W).
        """
        B, C, H, W = sample.shape

        # ── 1. Normalize and Format Timestep Tensor ──
        if not torch.is_tensor(timestep):
            timesteps = torch.tensor([timestep], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timestep) and len(timestep.shape) == 0:
            timesteps = timestep[None].to(sample.device)
        else:
            timesteps = timestep
        if timesteps.shape[0] != B:
            timesteps = timesteps.expand(B)

        # ── 2. Embed Diffusion Timestep into Conditioning Vector ──
        # c_t shape: (B, hidden_size)
        c_t = self.t_embedder(timesteps).to(dtype=sample.dtype)

        # ── 3. Distill Multimodal Story Context via Perceiver IO ──
        # Compresses variable-length tokens down to exactly M=64 distilled tokens: (B, 64, hidden_size)
        fused_context = self.perceiver(encoder_hidden_states, mask=encoder_attention_mask).to(dtype=sample.dtype)

        # ── 4. Apply Residual Pre-Convolutional Stem ──
        # Learns continuous local boundary filters prior to tokenization
        x = self.pre_conv(sample)

        # ── 5. Patchify & Add 2D Sinusoidal Spatial Positional Embedding ──
        # Maps (B, 4, H, W) -> (B, num_patches, hidden_size) e.g., (B, 1024, 768)
        x = self.patch_embed(x)
        num_patches = x.shape[1]

        # Resolution-Agnostic Positional Embedding:
        # If input resolution matches buffer, use directly; otherwise dynamically interpolate coordinates
        if self.pos_embed.shape[1] == num_patches:
            active_pos_embed = self.pos_embed.to(device=x.device, dtype=x.dtype)
        else:
            # Bicubic 2D position embedding interpolation for dynamic resolution changes (e.g. 384x384 -> 512x512)
            orig_grid = int(math.isqrt(self.pos_embed.shape[1]))
            new_h = H // self.patch_size
            new_w = W // self.patch_size
            pos_tokens = self.pos_embed.reshape(1, orig_grid, orig_grid, -1).permute(0, 3, 1, 2).float()
            new_pos_embed = F.interpolate(
                pos_tokens, size=(new_h, new_w), mode='bicubic', align_corners=False
            ).permute(0, 2, 3, 1).flatten(1, 2)
            active_pos_embed = new_pos_embed.to(device=x.device, dtype=x.dtype)

        # Inject initial Layer 0 positional coordinates
        x = x + active_pos_embed

        # ── 6. Process through 12 DiT Transformer Blocks ──
        # Uses Deep Positional Re-Anchoring (active_pos_embed injected at each block)
        # with optional activation checkpointing for VRAM efficiency
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                # Activation checkpointing: recomputes activations during backward pass to save VRAM
                x = torch.utils.checkpoint.checkpoint(
                    block, x, c_t, fused_context, active_pos_embed, use_reentrant=False
                )
            else:
                x = block(x, c_t, context=fused_context, pos_embed=active_pos_embed)

        # ── 7. Final Layer & Unpatchify ──
        # Modulates tokens with timestep adaLN, applies linear projection, and unpatchifies back to (B, 4, H, W)
        out = self.final_layer(x, c_t, H, W)

        # ── 8. Apply Residual Post-Convolutional Smoothing Head ──
        # Cancels high-frequency boundary artifacts in the predicted noise tensor
        out = self.post_conv(out)

        if not return_dict:
            return (out,)
        return PerceiverIODiTOutput(sample=out)
