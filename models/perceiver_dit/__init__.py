"""
PerceiverIO-DiT Module Package:
A modular Diffusion Transformer (DiT-B) with Perceiver IO multimodal conditioning
for Autoregressive Latent Diffusion Models (ARLDM).

Exported Architecture Components:
  - PerceiverIODiT: Drop-in replacement for UNet2DConditionModel in ARLDM.
  - PerceiverIODiTOutput: Dataclass container matching diffusers UNet output interface.
  - PerceiverIO: Multimodal context bottleneck resampler (condenses 385 tokens -> 64 tokens).
  - DiTBlock: Diffusion Transformer block with adaLN-Zero, self-attn, cross-attn, and deep positional anchoring.
  - PatchEmbed: 2D latent image to token sequence embedding.
  - TimestepEmbedder: Sinusoidal frequency embedder for diffusion timesteps.
  - PreConvStem: Residual 3x3 convolution for spatial continuity on VAE latents.
  - PostConvSmoothing: Residual 3x3 convolution for noise prediction smoothing.
  - FinalLayer: Final adaLN modulation, linear projection, and unpatchify head.
"""

from .fusion_model import PerceiverIODiT, PerceiverIODiTOutput
from .perceiver_io import PerceiverIO
from .dit import (
    DiTBlock,
    PatchEmbed,
    TimestepEmbedder,
    PreConvStem,
    PostConvSmoothing,
    FinalLayer,
)

__all__ = [
    "PerceiverIODiT",
    "PerceiverIODiTOutput",
    "PerceiverIO",
    "DiTBlock",
    "PatchEmbed",
    "TimestepEmbedder",
    "PreConvStem",
    "PostConvSmoothing",
    "FinalLayer",
]
