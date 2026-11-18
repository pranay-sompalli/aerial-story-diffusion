import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class PerceiverAttention(nn.Module):
    """
    Multi-Head Attention module supporting both Self-Attention and Cross-Attention
    using PyTorch's native scaled_dot_product_attention (SDPA).

    Key Characteristics:
      - When `context is None`: Operates as Self-Attention (Q, K, V all derived from x).
      - When `context is not None`: Operates as Cross-Attention (Q from x, K & V from context).
      - Utilizes native PyTorch SDPA (`F.scaled_dot_product_attention`), automatically
        leveraging FlashAttention or Memory-Efficient Attention kernels for optimal GPU efficiency.
    """
    def __init__(
        self,
        query_dim: int,
        context_dim: Optional[int] = None,
        heads: int = 8,
        head_dim: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        # inner_dim is the total dimension across all attention heads (e.g., 8 * 64 = 512)
        inner_dim = heads * head_dim
        # If context_dim is not specified, default to query_dim (self-attention configuration)
        context_dim = context_dim if context_dim is not None else query_dim

        # Linear projections for Query, Key, and Value
        # bias=False is standard practice to save parameters and match modern transformer norms
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        # Output projection to map multi-head representations back to query dimension
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N_q, query_dim) - Input query sequence (e.g., latent queries or patch tokens).
            context: (B, N_kv, context_dim) - Conditioning key/value sequence (or None for self-attention).
            mask: (B, N_kv) boolean mask where True indicates masked-out (padding) tokens,
                  or shaped (B, 1, N_q, N_kv) / (B, N_q, N_kv) attention mask.
        Returns:
            (B, N_q, query_dim) - Output sequence with updated attention-weighted features.
        """
        B, N_q, _ = x.shape
        # Check if cross-attention or self-attention is being performed
        is_cross = context is not None
        context = context if is_cross else x
        N_kv = context.shape[1]

        # 1. Project input tensors to Query, Key, and Value
        q = self.to_q(x)          # Shape: (B, N_q, inner_dim)
        k = self.to_k(context)    # Shape: (B, N_kv, inner_dim)
        v = self.to_v(context)    # Shape: (B, N_kv, inner_dim)

        # 2. Reshape and transpose into multi-head format: (B, heads, seq_len, head_dim)
        # Transposing places 'heads' at dim 1 so each attention head operates in parallel
        q = q.view(B, N_q, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N_kv, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N_kv, self.heads, self.head_dim).transpose(1, 2)

        # 3. Handle attention mask for PyTorch Scaled Dot Product Attention (SDPA)
        attn_mask = None
        if mask is not None:
            if mask.dtype == torch.bool:
                # In ARLDM dataloaders, boolean True denotes masked-out/padding tokens.
                # However, PyTorch SDPA expects boolean True for VALID tokens to attend to.
                # Therefore, we invert the mask (`~mask`) to get valid tokens.
                valid_mask = ~mask
                if valid_mask.ndim == 2:
                    # Expand (B, N_kv) -> (B, 1, 1, N_kv) to broadcast across heads and queries
                    attn_mask = valid_mask.unsqueeze(1).unsqueeze(2)
                elif valid_mask.ndim == 3:
                    # Expand (B, N_q, N_kv) -> (B, 1, N_q, N_kv) to broadcast across heads
                    attn_mask = valid_mask.unsqueeze(1)
                else:
                    attn_mask = valid_mask
            else:
                # Additive floating-point mask (e.g. -inf for masked positions)
                if mask.ndim == 2:
                    attn_mask = mask.unsqueeze(1).unsqueeze(2)
                elif mask.ndim == 3:
                    attn_mask = mask.unsqueeze(1)
                else:
                    attn_mask = mask

        # 4. Compute Flash / Memory-Efficient Attention via PyTorch native SDPA
        # Calculates: softmax((Q @ K^T) / sqrt(d_k) + mask) @ V
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
        )

        # 5. Concatenate heads back into a single feature vector per token
        # (B, heads, N_q, head_dim) -> (B, N_q, heads, head_dim) -> (B, N_q, inner_dim)
        out = out.transpose(1, 2).contiguous().view(B, N_q, self.heads * self.head_dim)

        # 6. Final linear projection back to query dimension
        return self.to_out(out)


class PerceiverMLP(nn.Module):
    """
    Feed-Forward Network (FFN) for Perceiver latent query processing.
    Uses GELU with tanh approximation and linear expansion.
    """
    def __init__(self, dim: int, mult: float = 4.0, dropout: float = 0.0):
        super().__init__()
        # Expand hidden dimension by mult (typically 4x) for nonlinear capacity
        inner_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(approximate="tanh"), # Approximate tanh GELU provides numerical stability
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),   # Project back to original latent dimension
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PerceiverLatentBlock(nn.Module):
    """
    Self-Attention + MLP Transformer block operating strictly on the fixed latent queries.
    Uses Pre-LayerNorm (norm before attention/MLP) to ensure smooth gradient flow through
    the residual highway.
    """
    def __init__(self, dim: int, heads: int = 8, head_dim: int = 64, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        # Pre-LayerNorm for Self-Attention
        self.norm1 = nn.LayerNorm(dim)
        # Multi-Head Self-Attention among latent queries (context=None)
        self.self_attn = PerceiverAttention(query_dim=dim, context_dim=None, heads=heads, head_dim=head_dim, dropout=dropout)
        # Pre-LayerNorm for MLP
        self.norm2 = nn.LayerNorm(dim)
        # Feed-forward MLP
        self.mlp = PerceiverMLP(dim=dim, mult=mlp_ratio, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm Self-Attention with residual connection
        x = x + self.self_attn(self.norm1(x))
        # Pre-norm MLP with residual connection
        x = x + self.mlp(self.norm2(x))
        return x


class PerceiverIO(nn.Module):
    """
    Perceiver IO Multimodal Conditioning Resampler:
    Distills variable-length, high-dimensional multimodal sequences (CLIP text tokens +
    BLIP past story frames across multiple timesteps) into a compact, fixed set of
    high-capacity latent tokens (M=64 tokens) via Cross-Attention and deep Latent Self-Attention.

    Why this is critical for ARLDM:
      - Variable-length multimodal story conditioning contains hundreds of tokens (5 frames x 77 tokens = 385 tokens).
      - Passing 385 tokens directly into all 12 DiT layers creates quadratic compute overhead in cross-attention.
      - Perceiver IO compresses these 385 variable tokens down to 64 fixed semantic tokens, reducing DiT
        cross-attention compute by ~83% while distilling cross-modal reasoning into a fixed query manifold.
    """
    def __init__(
        self,
        num_latents: int = 64,       # Number of learned bottleneck latent queries (M=64)
        latent_dim: int = 768,       # Dimension of Perceiver latent space
        context_dim: int = 768,      # Dimension of input multimodal context (CLIP/BLIP features)
        out_dim: Optional[int] = None, # Output dimension for DiT cross-attention (defaults to latent_dim)
        depth: int = 2,              # Number of latent self-attention processing layers
        heads: int = 8,              # Attention heads for Perceiver attention
        head_dim: Optional[int] = None, # Dimension per attention head
        mlp_ratio: float = 4.0,      # MLP expansion ratio
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_latents = num_latents
        self.latent_dim = latent_dim
        self.out_dim = out_dim if out_dim is not None else latent_dim
        head_dim = head_dim if head_dim is not None else (latent_dim // heads)

        # Learnable latent query vectors: (num_latents, latent_dim).
        # Initialized with small normal variance (std=0.02) to avoid saturating early attention.
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim) * 0.02)

        # 1. Cross-attention module: queries attend to the full multimodal context
        self.norm_latents = nn.LayerNorm(latent_dim)
        self.norm_context = nn.LayerNorm(context_dim)
        self.cross_attn = PerceiverAttention(
            query_dim=latent_dim,
            context_dim=context_dim,
            heads=heads,
            head_dim=head_dim,
            dropout=dropout,
        )
        self.cross_mlp = PerceiverMLP(dim=latent_dim, mult=mlp_ratio, dropout=dropout)
        self.norm_cross_mlp = nn.LayerNorm(latent_dim)

        # 2. Latent self-attention processing layers: cross-modal reasoning among distilled queries
        self.layers = nn.ModuleList([
            PerceiverLatentBlock(
                dim=latent_dim,
                heads=heads,
                head_dim=head_dim,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(depth)
        ])

        # Final normalization before output projection
        self.final_norm = nn.LayerNorm(latent_dim)

        # Output projection if DiT hidden dimension differs from Perceiver latent dimension
        if self.out_dim != latent_dim:
            self.proj_out = nn.Linear(latent_dim, self.out_dim)
        else:
            self.proj_out = nn.Identity()

    def forward(
        self,
        context: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            context: (B, seq_len, context_dim) - Multimodal tokens (CLIP text + BLIP frame history).
            mask: (B, seq_len) - Boolean mask where True indicates masked-out / padding tokens.
        Returns:
            fused_latents: (B, num_latents, out_dim) - Exactly 64 distilled multimodal tokens.
        """
        B = context.shape[0]

        # Expand the learnable query parameters across the batch dimension:
        # (num_latents, latent_dim) -> (1, num_latents, latent_dim) -> (B, num_latents, latent_dim)
        latents = self.latents.unsqueeze(0).expand(B, -1, -1)

        # Step 1: Cross-attention - Distill input multimodal context into the latent query representations
        normed_latents = self.norm_latents(latents)
        normed_context = self.norm_context(context)
        latents = latents + self.cross_attn(normed_latents, normed_context, mask=mask)
        latents = latents + self.cross_mlp(self.norm_cross_mlp(latents))

        # Step 2: Deep latent self-attention - Perform internal cross-modal reasoning and feature binding
        for layer in self.layers:
            latents = layer(latents)

        # Step 3: Final LayerNorm and projection to DiT context dimension
        latents = self.final_norm(latents)
        return self.proj_out(latents)
