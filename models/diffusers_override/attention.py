# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.embeddings import ImagePositionalEmbeddings
from diffusers.utils import BaseOutput
from diffusers.utils.import_utils import is_xformers_available


@dataclass
class Transformer2DModelOutput(BaseOutput):
    """
    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` or `(batch size, num_vector_embeds - 1, num_latent_pixels)` if [`Transformer2DModel`] is discrete):
            Hidden states conditioned on `encoder_hidden_states` input. If discrete, returns probability distributions
            for the unnoised latent pixels.
    """

    sample: torch.FloatTensor


if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None


class Transformer2DModel(ModelMixin, ConfigMixin):
    """
    Transformer model for image-like data. Takes either discrete (classes of vector embeddings) or continuous (actual
    embeddings) inputs.

    When input is continuous: First, project the input (aka embedding) and reshape to b, t, d. Then apply standard
    transformer action. Finally, reshape to image.

    When input is discrete: First, input (classes of latent pixels) is converted to embeddings and has positional
    embeddings applied, see `ImagePositionalEmbeddings`. Then apply standard transformer action. Finally, predict
    classes of unnoised image.

    Note that it is assumed one of the input classes is the masked latent pixel. The predicted classes of the unnoised
    image do not contain a prediction for the masked pixel as the unnoised image cannot be masked.

    Parameters:
        num_attention_heads (`int`, *optional*, defaults to 16): The number of heads to use for multi-head attention.
        attention_head_dim (`int`, *optional*, defaults to 88): The number of channels in each head.
        in_channels (`int`, *optional*):
            Pass if the input is continuous. The number of channels in the input and output.
        num_layers (`int`, *optional*, defaults to 1): The number of layers of Transformer blocks to use.
        dropout (`float`, *optional*, defaults to 0.1): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The number of context dimensions to use.
        sample_size (`int`, *optional*): Pass if the input is discrete. The width of the latent images.
            Note that this is fixed at training time as it is used for learning a number of position embeddings. See
            `ImagePositionalEmbeddings`.
        num_vector_embeds (`int`, *optional*):
            Pass if the input is discrete. The number of classes of the vector embeddings of the latent pixels.
            Includes the class for the masked latent pixel.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        num_embeds_ada_norm ( `int`, *optional*): Pass if at least one of the norm_layers is `AdaLayerNorm`.
            The number of diffusion steps used during training. Note that this is fixed at training time as it is used
            to learn a number of embeddings that are added to the hidden states. During inference, you can denoise for
            up to but not more than steps than `num_embeds_ada_norm`.
        attention_bias (`bool`, *optional*):
            Configure if the TransformerBlocks' attention should contain a bias parameter.
    """

    @register_to_config
    def __init__(
            self,
            num_attention_heads: int = 16,
            attention_head_dim: int = 88,
            in_channels: Optional[int] = None,
            num_layers: int = 1,
            dropout: float = 0.0,
            norm_num_groups: int = 32,
            cross_attention_dim: Optional[int] = None,
            attention_bias: bool = False,
            sample_size: Optional[int] = None,
            num_vector_embeds: Optional[int] = None,
            activation_fn: str = "geglu",
            num_embeds_ada_norm: Optional[int] = None,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim

        # Differing from the original Transformer2DModel: this version supports both continuous image tensors
        # and discrete/vectorized inputs (for VQ-VAE style models). The input type determines how we project.
        self.is_input_continuous = in_channels is not None
        self.is_input_vectorized = num_vector_embeds is not None

        if self.is_input_continuous and self.is_input_vectorized:
            raise ValueError(
                f"Cannot define both `in_channels`: {in_channels} and `num_vector_embeds`: {num_vector_embeds}. Make"
                " sure that either `in_channels` or `num_vector_embeds` is None."
            )
        elif not self.is_input_continuous and not self.is_input_vectorized:
            raise ValueError(
                f"Has to define either `in_channels`: {in_channels} or `num_vector_embeds`: {num_vector_embeds}. Make"
                " sure that either `in_channels` or `num_vector_embeds` is not None."
            )

        # Handle input projection for continuous images: normalize then project to inner dimension.
        # This differs from the standard transformer which typically doesn't include GroupNorm.
        if self.is_input_continuous:
            self.in_channels = in_channels

            self.norm = torch.nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True)
            self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)
        elif self.is_input_vectorized:
            # For discrete/vectorized input, we need positional embeddings to encode spatial location.
            assert sample_size is not None, "Transformer2DModel over discrete input must provide sample_size"
            assert num_vector_embeds is not None, "Transformer2DModel over discrete input must provide num_embed"

            self.height = sample_size
            self.width = sample_size
            self.num_vector_embeds = num_vector_embeds
            self.num_latent_pixels = self.height * self.width

            self.latent_image_embedding = ImagePositionalEmbeddings(
                num_embed=num_vector_embeds, embed_dim=inner_dim, height=self.height, width=self.width
            )

        # Create transformer blocks that will process the input sequentially.
        # Differs from standard transformers: uses GEGLU activation by default and supports AdaLayerNorm for time steps.
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    num_embeds_ada_norm=num_embeds_ada_norm,
                    attention_bias=attention_bias,
                )
                for d in range(num_layers)
            ]
        )

        # Output projection: map back to input space (continuous) or to vocab predictions (vectorized).
        # Unlike the original which only has projection for continuous input, this supports both.
        if self.is_input_continuous:
            self.proj_out = nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)
        elif self.is_input_vectorized:
            self.norm_out = nn.LayerNorm(inner_dim)
            self.out = nn.Linear(inner_dim, self.num_vector_embeds - 1)

    def _set_attention_slice(self, slice_size):
        for block in self.transformer_blocks:
            block._set_attention_slice(slice_size)

    def forward(self, hidden_states, encoder_hidden_states=None, encoder_attention_mask=None, timestep=None,
                return_dict: bool = True):
        """
        Args:
            hidden_states ( When discrete, `torch.LongTensor` of shape `(batch size, num latent pixels)`.
                When continous, `torch.FloatTensor` of shape `(batch size, channel, height, width)`): Input
                hidden_states
            encoder_hidden_states ( `torch.LongTensor` of shape `(batch size, context dim)`, *optional*):
                Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
                self-attention.
            encoder_attention_mask ( `torch.LongTensor` of shape `(batch size, context)`, *optional*):
                Attention mask for cross attention layer.
            timestep ( `torch.long`, *optional*):
                Optional timestep to be applied as an embedding in AdaLayerNorm's. Used to indicate denoising step.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain tuple.

        Returns:
            [`~models.attention.Transformer2DModelOutput`] or `tuple`: [`~models.attention.Transformer2DModelOutput`]
            if `return_dict` is True, otherwise a `tuple`. When returning a tuple, the first element is the sample
            tensor.
        """
        # 1. Input: reshape image tensor to sequence format (B, H*W, C) for transformer processing.
        # This differs from text transformers which already have sequence format; images need reshaping.
        if self.is_input_continuous:
            batch, channel, height, weight = hidden_states.shape
            residual = hidden_states
            hidden_states = self.norm(hidden_states)
            hidden_states = self.proj_in(hidden_states)
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * weight, inner_dim)
        elif self.is_input_vectorized:
            # For discrete inputs, embed the class tokens and add positional embeddings.
            hidden_states = self.latent_image_embedding(hidden_states)

        # 2. Process through transformer blocks.
        # Differs from standard transformers: each block handles both self-attention and optional cross-attention.
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, context=encoder_hidden_states, mask=encoder_attention_mask,
                                  timestep=timestep)

        # 3. Output: reshape back to image format and add skip connection.
        # For vectorized inputs, we compute log probabilities over the vocabulary.
        if self.is_input_continuous:
            hidden_states = hidden_states.reshape(batch, height, weight, inner_dim).permute(0, 3, 1, 2)
            hidden_states = self.proj_out(hidden_states)
            output = hidden_states + residual
        elif self.is_input_vectorized:
            hidden_states = self.norm_out(hidden_states)
            logits = self.out(hidden_states)
            # (batch, self.num_vector_embeds - 1, self.num_latent_pixels)
            logits = logits.permute(0, 2, 1)

            # log(p(x_0)) - compute log probabilities for discrete diffusion models
            output = F.log_softmax(logits.double(), dim=1).float()

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

    def _set_use_memory_efficient_attention_xformers(self, use_memory_efficient_attention_xformers: bool):
        for block in self.transformer_blocks:
            block._set_use_memory_efficient_attention_xformers(use_memory_efficient_attention_xformers)


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other. Originally ported from here, but adapted
    to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    Uses three q, k, v linear layers to compute attention.

    Differs from the original transformer attention: includes GroupNorm preprocessing (instead of just LayerNorm)
    and an optional rescaling factor to stabilize training in diffusion models.
    """

    def __init__(
            self,
            channels: int,
            num_head_channels: Optional[int] = None,
            norm_num_groups: int = 32,
            rescale_output_factor: float = 1.0,
            eps: float = 1e-5,
    ):
        super().__init__()
        self.channels = channels

        # Compute number of heads from channel count and head size.
        # Differing from standard multi-head attention: uses a more flexible head dimension calculation.
        self.num_heads = channels // num_head_channels if num_head_channels is not None else 1
        self.num_head_size = num_head_channels
        self.group_norm = nn.GroupNorm(num_channels=channels, num_groups=norm_num_groups, eps=eps, affine=True)

        # Query, key, value projection layers - standard attention mechanism.
        self.query = nn.Linear(channels, channels)
        self.key = nn.Linear(channels, channels)
        self.value = nn.Linear(channels, channels)

        self.rescale_output_factor = rescale_output_factor
        # Output projection with optional rescaling for stable training.
        self.proj_attn = nn.Linear(channels, channels, 1)

    def transpose_for_scores(self, projection: torch.Tensor) -> torch.Tensor:
        new_projection_shape = projection.size()[:-1] + (self.num_heads, -1)
        # move heads to 2nd position (B, T, H * D) -> (B, T, H, D) -> (B, H, T, D)
        new_projection = projection.view(new_projection_shape).permute(0, 2, 1, 3)
        return new_projection

    def forward(self, hidden_states):
        # Residual connection is added at the end for gradient flow.
        residual = hidden_states
        batch, channel, height, width = hidden_states.shape

        # Normalize input with GroupNorm - differs from standard attention which uses LayerNorm or no norm.
        # This helps stabilize training in diffusion UNet architectures.
        hidden_states = self.group_norm(hidden_states)

        # Reshape: (B, C, H, W) -> (B, H*W, C) for sequence processing.
        # This makes spatial positions attend to each other.
        hidden_states = hidden_states.view(batch, channel, height * width).transpose(1, 2)

        # Project to query, key, value vectors via linear layers.
        query_proj = self.query(hidden_states)
        key_proj = self.key(hidden_states)
        value_proj = self.value(hidden_states)

        # Reshape to separate heads: (B, N, C) -> (B, H, N, D) where H=num_heads, D=head_dim.
        query_states = self.transpose_for_scores(query_proj)
        key_states = self.transpose_for_scores(key_proj)
        value_states = self.transpose_for_scores(value_proj)

        # Compute attention scores with an unusual scaling: uses sqrt(sqrt()) instead of just sqrt().
        # This differs from standard scaled dot-product attention scaling.
        scale = 1 / math.sqrt(math.sqrt(self.channels / self.num_heads))
        attention_scores = torch.matmul(query_states * scale, key_states.transpose(-1, -2) * scale)
        attention_probs = torch.softmax(attention_scores.float(), dim=-1).type(attention_scores.dtype)

        # Apply attention to values to get the output.
        hidden_states = torch.matmul(attention_probs, value_states)

        # Reshape: (B, H, N, D) -> (B, N, C) to restore original shape.
        hidden_states = hidden_states.permute(0, 2, 1, 3).contiguous()
        new_hidden_states_shape = hidden_states.size()[:-2] + (self.channels,)
        hidden_states = hidden_states.view(new_hidden_states_shape)

        # Final linear projection.
        hidden_states = self.proj_attn(hidden_states)
        hidden_states = hidden_states.transpose(-1, -2).reshape(batch, channel, height, width)

        # Residual connection with rescaling - this rescaling factor helps with training stability.
        hidden_states = (hidden_states + residual) / self.rescale_output_factor
        return hidden_states


class BasicTransformerBlock(nn.Module):
    r"""
    A basic Transformer block.

    Differs from the original transformer: includes two attention layers (self-attention + optional cross-attention)
    and uses AdaLayerNorm when num_embeds_ada_norm is provided, allowing the normalization to be conditioned
    on the diffusion timestep. This is key for diffusion models where the denoiser needs to know the noise level.
    """

    def __init__(
            self,
            dim: int,
            num_attention_heads: int,
            attention_head_dim: int,
            dropout=0.0,
            cross_attention_dim: Optional[int] = None,
            activation_fn: str = "geglu",
            num_embeds_ada_norm: Optional[int] = None,
            attention_bias: bool = False,
    ):
        super().__init__()
        # Self-attention: allows each position to attend to all other positions.
        # Differing from standard transformers: uses custom CrossAttention that supports memory-efficient variants.
        self.attn1 = CrossAttention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
        )
        # Feed-forward network with GEGLU activation (or GELU approximate).
        # GEGLU differs from standard transformers which use GELU or ReLU.
        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn)
        # Cross-attention: allows attending to encoder hidden states (e.g., text embeddings in text-to-image).
        # When context is None, this becomes self-attention again.
        self.attn2 = CrossAttention(
            query_dim=dim,
            cross_attention_dim=cross_attention_dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
        )

        # Layer norms: either AdaLayerNorm (timestep-conditioned) or standard LayerNorm.
        # AdaLayerNorm is what makes this block suited for diffusion models.
        self.use_ada_layer_norm = num_embeds_ada_norm is not None
        if self.use_ada_layer_norm:
            self.norm1 = AdaLayerNorm(dim, num_embeds_ada_norm)
            self.norm2 = AdaLayerNorm(dim, num_embeds_ada_norm)
        else:
            self.norm1 = nn.LayerNorm(dim)
            self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def _set_attention_slice(self, slice_size):
        self.attn1._slice_size = slice_size
        self.attn2._slice_size = slice_size

    def _set_use_memory_efficient_attention_xformers(self, use_memory_efficient_attention_xformers: bool):
        if not is_xformers_available():
            print("Here is how to install it")
            raise ModuleNotFoundError(
                "Refer to https://github.com/facebookresearch/xformers for more information on how to install"
                " xformers",
                name="xformers",
            )
        elif not torch.cuda.is_available():
            raise ValueError(
                "torch.cuda.is_available() should be True but is False. xformers' memory efficient attention is only"
                " available for GPU "
            )
        else:
            try:
                # Make sure we can run the memory efficient attention
                _ = xformers.ops.memory_efficient_attention(
                    torch.randn((1, 2, 40), device="cuda"),
                    torch.randn((1, 2, 40), device="cuda"),
                    torch.randn((1, 2, 40), device="cuda"),
                )
            except Exception as e:
                raise e
            self.attn1._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers
            self.attn2._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers

    def forward(self, hidden_states, context=None, mask=None, timestep=None):
        # 1. Self-Attention with residual connection.
        # The timestep is passed to AdaLayerNorm to condition the normalization.
        norm_hidden_states = (
            self.norm1(hidden_states, timestep) if self.use_ada_layer_norm else self.norm1(hidden_states)
        )
        hidden_states = self.attn1(norm_hidden_states) + hidden_states

        # 2. Cross-Attention with residual connection.
        # Allows the model to attend to context (e.g., text embeddings) if provided.
        norm_hidden_states = (
            self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
        )
        hidden_states = self.attn2(norm_hidden_states, context=context, mask=mask) + hidden_states

        # 3. Feed-forward with residual connection.
        # Uses GEGLU or ApproximateGELU activation instead of standard GELU.
        hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states

        return hidden_states


class CrossAttention(nn.Module):
    r"""
    A cross attention layer.

    Differs from the original transformer cross-attention: supports memory-efficient attention variants
    (xformers) and sliced attention for large batch sizes, which are essential optimizations for
    diffusion model training.
    """

    def __init__(
            self,
            query_dim: int,
            cross_attention_dim: Optional[int] = None,
            heads: int = 8,
            dim_head: int = 64,
            dropout: float = 0.0,
            bias=False,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim

        # Scale factor for attention scores - standard 1/sqrt(dim_head) scaling.
        self.scale = dim_head ** -0.5
        self.heads = heads
        # These flags enable memory-efficient attention optimizations.
        # _slice_size splits attention computation across batch axis to save memory.
        self._slice_size = None
        self._use_memory_efficient_attention_xformers = False

        # Query, key, value linear projections.
        self.to_q = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_k = nn.Linear(cross_attention_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(cross_attention_dim, inner_dim, bias=bias)

        # Output projection with dropout for regularization.
        self.to_out = nn.ModuleList([])
        self.to_out.append(nn.Linear(inner_dim, query_dim))
        self.to_out.append(nn.Dropout(dropout))

    def reshape_heads_to_batch_dim(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size * head_size, seq_len, dim // head_size)
        return tensor

    def reshape_batch_dim_to_heads(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
        return tensor

    def forward(self, hidden_states, context=None, mask=None):
        batch_size, sequence_length, _ = hidden_states.shape

        # Project to query, key, value. Context defaults to hidden_states for self-attention.
        query = self.to_q(hidden_states)
        context = context if context is not None else hidden_states
        key = self.to_k(context)
        value = self.to_v(context)

        dim = query.shape[-1]

        # Reshape for efficient multi-head attention: batch dims combined so each head is processed independently.
        query = self.reshape_heads_to_batch_dim(query)
        key = self.reshape_heads_to_batch_dim(key)
        value = self.reshape_heads_to_batch_dim(value)

        # Expand mask across heads for broadcasting.
        mask = mask.repeat_interleave(self.heads, dim=0).unsqueeze(1) if mask is not None else None

        # Compute attention using one of three methods:
        # 1. xformers: memory-efficient external library for large-scale attention
        # 2. SDPA: PyTorch's built-in scaled dot-product attention (most efficient)
        # 3. Sliced: splits batch dimension to save memory when needed
        if self._use_memory_efficient_attention_xformers:
            hidden_states = self._memory_efficient_attention_xformers(query, key, value)
        else:
            if self._slice_size is None or query.shape[0] // self._slice_size == 1:
                hidden_states = self._attention(query, key, value, mask)
            else:
                assert mask is None, "masking is not supported for sliced attention"
                hidden_states = self._sliced_attention(query, key, value, sequence_length, dim)

        # Project back to original dimension and apply dropout.
        hidden_states = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)
        return hidden_states

    def _attention(self, query, key, value, mask):
        # Try to use PyTorch's optimized scaled dot-product attention (SDPA) if available (PyTorch 2.0+).
        # SDPA is more memory-efficient and often faster than manual implementation.
        if hasattr(F, "scaled_dot_product_attention"):
            # SDPA expects mask where True means "keep" (attend to), but our mask has True for "mask out".
            # We invert the mask to match SDPA's convention.
            attn_mask = ~mask if mask is not None else None

            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
            )
        else:
            # Fallback implementation for older PyTorch versions.
            # Uses einsum on MPS (Metal) for better performance on Apple Silicon GPUs.
            if query.device.type == "mps":
                attention_scores = torch.einsum("b i d, b j d -> b i j", query, key) * self.scale
            else:
                attention_scores = torch.matmul(query, key.transpose(-1, -2)) * self.scale
            attention_scores = attention_scores.masked_fill(mask.expand(attention_scores.shape), value=float("-inf")) \
                if mask is not None else attention_scores
            attention_probs = attention_scores.softmax(dim=-1)

            if query.device.type == "mps":
                hidden_states = torch.einsum("b i j, b j d -> b i d", attention_probs, value)
            else:
                hidden_states = torch.matmul(attention_probs, value)

        # Reshape from (B*H, N, D) back to (B, N, C) for the next layer.
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states

    def _sliced_attention(self, query, key, value, sequence_length, dim):
        # When batch size is too large for memory, process attention in slices.
        # This trades compute time for memory efficiency.
        batch_size_attention = query.shape[0]
        hidden_states = torch.zeros(
            (batch_size_attention, sequence_length, dim // self.heads), device=query.device, dtype=query.dtype
        )
        slice_size = self._slice_size if self._slice_size is not None else hidden_states.shape[0]
        for i in range(hidden_states.shape[0] // slice_size):
            start_idx = i * slice_size
            end_idx = (i + 1) * slice_size
            if query.device.type == "mps":
                # MPS devices (Apple Silicon) perform better with einsum.
                attn_slice = (
                        torch.einsum("b i d, b j d -> b i j", query[start_idx:end_idx], key[start_idx:end_idx])
                        * self.scale
                )
            else:
                attn_slice = (
                        torch.matmul(query[start_idx:end_idx], key[start_idx:end_idx].transpose(1, 2)) * self.scale
                )
            attn_slice = attn_slice.softmax(dim=-1)
            if query.device.type == "mps":
                attn_slice = torch.einsum("b i j, b j d -> b i d", attn_slice, value[start_idx:end_idx])
            else:
                attn_slice = torch.matmul(attn_slice, value[start_idx:end_idx])

            hidden_states[start_idx:end_idx] = attn_slice

        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states

    def _memory_efficient_attention_xformers(self, query, key, value):
        # Use xformers library for memory-efficient attention on very large models.
        hidden_states = xformers.ops.memory_efficient_attention(query, key, value, attn_bias=None)
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states


class FeedForward(nn.Module):
    r"""
    A feed-forward layer.

    Differs from the original transformer FFN: uses GEGLU or ApproximateGELU instead of GELU,
    which have been shown to work better for diffusion model training stability.
    """

    def __init__(
            self,
            dim: int,
            dim_out: Optional[int] = None,
            mult: int = 4,
            dropout: float = 0.0,
            activation_fn: str = "geglu",
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        # Choose activation: GEGLU (gated linear unit) or ApproximateGELU.
        # GEGLU doubles the hidden dimension and uses a learned gate, differing from standard GELU.
        if activation_fn == "geglu":
            geglu = GEGLU(dim, inner_dim)
        elif activation_fn == "geglu-approximate":
            geglu = ApproximateGELU(dim, inner_dim)

        self.net = nn.ModuleList([])
        self.net.append(geglu)
        self.net.append(nn.Dropout(dropout))
        self.net.append(nn.Linear(inner_dim, dim_out))

    def forward(self, hidden_states):
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


# feedforward
class GEGLU(nn.Module):
    r"""
    A variant of the gated linear unit activation function from https://arxiv.org/abs/2002.05202.

    Differs from standard GELU: splits projection in half and uses one half as a gate for the other,
    creating a learnable gating mechanism that can improve training dynamics in diffusion models.
    """

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        # Double the output dimension to split into value and gate projections.
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def gelu(self, gate):
        # MPS (Metal) devices need float32 for GELU, then convert back.
        if gate.device.type != "mps":
            return F.gelu(gate)
        return F.gelu(gate.to(dtype=torch.float32)).to(dtype=gate.dtype)

    def forward(self, hidden_states):
        # Split the projection output into value and gate, apply GELU to gate.
        hidden_states, gate = self.proj(hidden_states).chunk(2, dim=-1)
        return hidden_states * self.gelu(gate)


class ApproximateGELU(nn.Module):
    """
    The approximate form of Gaussian Error Linear Unit (GELU).

    Differs from standard GELU: uses sigmoid approximation x * sigmoid(1.702 * x) which is faster
    to compute and was used in earlier transformer implementations.
    """

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out)

    def forward(self, x):
        x = self.proj(x)
        # Approximate GELU using sigmoid: faster than exact GELU, often used in early transformers.
        return x * torch.sigmoid(1.702 * x)


class AdaLayerNorm(nn.Module):
    """
    Norm layer modified to incorporate timestep embeddings.
    Differs from standard LayerNorm: the scale and shift parameters are dynamically computed
    from the timestep embedding, allowing the normalization to be conditioned on the diffusion step.
    """

    def __init__(self, embedding_dim, num_embeddings):
        super().__init__()
        # Embedding that maps timestep indices to vectors.
        self.emb = nn.Embedding(num_embeddings, embedding_dim)
        # Use SiLU (Swish) activation to create smoother timestep embeddings.
        self.silu = nn.SiLU()
        # Project to get both scale and shift parameters from the embedding.
        self.linear = nn.Linear(embedding_dim, embedding_dim * 2)
        # LayerNorm without learnable affine parameters - we'll provide them dynamically.
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False)

    def forward(self, x, timestep):
        # Compute scale and shift from timestep embedding.
        emb = self.linear(self.silu(self.emb(timestep)))
        scale, shift = torch.chunk(emb, 2)
        # Apply LayerNorm then modulate with the computed scale and shift.
        x = self.norm(x) * (1 + scale) + shift
        return x
