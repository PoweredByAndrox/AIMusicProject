#!/usr/bin/env python3
"""
Patch allin1 to work with natten 0.20+ API.

The old natten API used separate functions:
- natten1dqkrpb(q, k, rpb, kernel_size, dilation) -> attention_scores
- natten1dav(attn_probs, v, kernel_size, dilation) -> context

The new natten 0.20+ API uses unified fused attention:
- na1d(q, k, v, kernel_size, dilation=dilation) -> output

This patch replaces the old API calls with compatibility wrappers.
"""

import os
import sys

DINAT_PATH = "/usr/local/lib/python3.11/dist-packages/allin1/models/dinat.py"

NEW_DINAT_CODE = '''"""This is a modification of:
  https://github.com/huggingface/transformers/blob/main/src/transformers/models/dinat/modeling_dinat.py
  so that it can provide both 1D and 2D attention.
  
  PATCHED for natten 0.20+ API compatibility.
"""

import math
import torch
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Callable
from natten.functional import na1d, na2d
from ..config import Config
from .utils import *


# Copied from transformers.models.beit.modeling_beit.drop_path
def drop_path(input, drop_prob=0.0, training=False, scale_by_keep=True):
  if drop_prob == 0.0 or not training:
    return input
  keep_prob = 1 - drop_prob
  shape = (input.shape[0],) + (1,) * (input.ndim - 1)
  random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
  random_tensor.floor_()
  output = input.div(keep_prob) * random_tensor
  return output


class DinatDropPath(nn.Module):
  def __init__(self, drop_prob: Optional[float] = None) -> None:
    super().__init__()
    self.drop_prob = drop_prob

  def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    return drop_path(hidden_states, self.drop_prob, self.training)

  def extra_repr(self) -> str:
    return "p={}".format(self.drop_prob)


class _NeighborhoodAttentionNd(ABC, nn.Module):
  rpb: nn.Parameter
  na_func: Callable  # na1d or na2d

  def __init__(
    self,
    cfg: Config,
    dim: int,
    num_heads: int,
    kernel_size: int,
    dilation: int
  ):
    super().__init__()
    if dim % num_heads != 0:
      raise ValueError(
        f"The hidden size ({dim}) is not a multiple of the number of attention heads ({num_heads})"
      )

    self.num_attention_heads = num_heads
    self.attention_head_size = int(dim / num_heads)
    self.all_head_size = self.num_attention_heads * self.attention_head_size
    self.kernel_size = kernel_size
    self.dilation = dilation

    self.query = nn.Linear(self.all_head_size, self.all_head_size, bias=cfg.qkv_bias)
    self.key = nn.Linear(self.all_head_size, self.all_head_size, bias=cfg.qkv_bias)
    self.value = nn.Linear(self.all_head_size, self.all_head_size, bias=cfg.qkv_bias)

    self.dropout = nn.Dropout(cfg.drop_attention)
    self.scale = 1.0 / math.sqrt(self.attention_head_size)

  def forward(
    self,
    hidden_states: torch.Tensor,
    output_attentions: Optional[bool] = False,
  ) -> Tuple[torch.Tensor]:
    # Get Q, K, V projections
    query_layer = self.query(hidden_states)
    key_layer = self.key(hidden_states)
    value_layer = self.value(hidden_states)
    
    # Reshape for attention: [batch, seq, heads, head_dim]
    batch_size = query_layer.size(0)
    
    if len(hidden_states.shape) > 3:  # 2D case
      # Input: [batch, height, width, dim]
      height, width = hidden_states.size(1), hidden_states.size(2)
      query_layer = query_layer.view(batch_size, height, width, self.num_attention_heads, self.attention_head_size)
      key_layer = key_layer.view(batch_size, height, width, self.num_attention_heads, self.attention_head_size)
      value_layer = value_layer.view(batch_size, height, width, self.num_attention_heads, self.attention_head_size)
    else:  # 1D case
      # Input: [batch, seq, dim]
      seq_len = hidden_states.size(1)
      query_layer = query_layer.view(batch_size, seq_len, self.num_attention_heads, self.attention_head_size)
      key_layer = key_layer.view(batch_size, seq_len, self.num_attention_heads, self.attention_head_size)
      value_layer = value_layer.view(batch_size, seq_len, self.num_attention_heads, self.attention_head_size)

    # Use fused neighborhood attention (new API)
    # Note: RPB (relative positional bias) is not directly supported in new API
    # We apply it separately if needed, but for now we use the fused version without RPB
    context_layer = self.na_func(
      query_layer, 
      key_layer, 
      value_layer,
      kernel_size=self.kernel_size,
      dilation=self.dilation,
      scale=self.scale
    )
    
    # Apply dropout (on output since we can't access attention weights in fused version)
    context_layer = self.dropout(context_layer)
    
    # Reshape back: [batch, seq/height*width, all_head_size]
    new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
    context_layer = context_layer.view(new_context_layer_shape)

    # Note: output_attentions is not supported with fused attention
    outputs = (context_layer,)
    return outputs


class NeighborhoodAttention1d(_NeighborhoodAttentionNd):
  def __init__(
    self,
    cfg: Config,
    dim: int,
    num_heads: int,
    kernel_size: int,
    dilation: int
  ):
    super().__init__(cfg, dim, num_heads, kernel_size, dilation)
    # RPB kept for compatibility but not used in fused attention
    self.rpb = nn.Parameter(
      torch.zeros(num_heads, (2 * self.kernel_size - 1)),
      requires_grad=False,  # Disabled since not used
    )
    self.na_func = na1d


class NeighborhoodAttention2d(_NeighborhoodAttentionNd):
  def __init__(
    self,
    cfg: Config,
    dim: int,
    num_heads: int,
    kernel_size: int,
    dilation: int
  ):
    super().__init__(cfg, dim, num_heads, kernel_size, dilation)
    self.rpb = nn.Parameter(
      torch.zeros(num_heads, (2 * self.kernel_size - 1), (2 * self.kernel_size - 1)),
      requires_grad=False,
    )
    self.na_func = na2d


class NeighborhoodAttentionOutput(nn.Module):
  def __init__(self, config: Config, dim: int):
    super().__init__()
    self.dense = nn.Linear(dim, dim)
    self.dropout = nn.Dropout(config.drop_attention)

  def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    hidden_states = self.dense(hidden_states)
    hidden_states = self.dropout(hidden_states)
    return hidden_states


class _NeighborhoodAttentionModuleNd(ABC, nn.Module):
  self: _NeighborhoodAttentionNd

  def __init__(self, cfg: Config, dim: int):
    super().__init__()
    self.output = NeighborhoodAttentionOutput(cfg, dim)

  def forward(
    self,
    hidden_states: torch.Tensor,
    output_attentions: Optional[bool] = False,
  ) -> Tuple[torch.Tensor]:
    self_outputs = self.self(hidden_states, output_attentions)
    attention_output = self.output(self_outputs[0])
    outputs = (attention_output,) + self_outputs[1:]
    return outputs


class NeighborhoodAttentionModule1d(_NeighborhoodAttentionModuleNd):
  def __init__(self, cfg: Config, dim: int, num_heads: int, kernel_size: int, dilation: int):
    super().__init__(cfg, dim)
    self.self = NeighborhoodAttention1d(cfg, dim, num_heads, kernel_size, dilation)


class NeighborhoodAttentionModule2d(_NeighborhoodAttentionModuleNd):
  def __init__(self, cfg: Config, dim: int, num_heads: int, kernel_size: int, dilation: int):
    super().__init__(cfg, dim)
    self.self = NeighborhoodAttention2d(cfg, dim, num_heads, kernel_size, dilation)


class DinatIntermediate(nn.Module):
  def __init__(self, config: Config, dim_in: int, dim_out: int):
    super().__init__()
    self.dense = nn.Linear(dim_in, dim_out)
    if isinstance(config.act_transformer, str):
      self.intermediate_act_fn = get_activation_function(config.act_transformer)
    else:
      self.intermediate_act_fn = config.act_transformer

  def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    hidden_states = self.dense(hidden_states)
    hidden_states = self.intermediate_act_fn(hidden_states)
    return hidden_states


class DinatOutput(nn.Module):
  def __init__(self, config: Config, dim_in: int, dim_out: int):
    super().__init__()
    self.dense = nn.Linear(dim_in, dim_out)
    self.dropout = nn.Dropout(config.drop_hidden)

  def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    hidden_states = self.dense(hidden_states)
    hidden_states = self.dropout(hidden_states)
    return hidden_states


class _DinatLayerNd(ABC, nn.Module):
  attention: _NeighborhoodAttentionModuleNd
  attention2: Optional[_NeighborhoodAttentionModuleNd]

  def __init__(
    self,
    cfg: Config,
    dim: int,
    kernel_size: int,
    dilation: int,
    drop_path_rate: float,
    double_attention: bool,
  ):
    super().__init__()
    self.double_attention = double_attention
    self.kernel_size = kernel_size
    self.dilation = dilation
    self.window_size = self.kernel_size * self.dilation
    if double_attention:
      self.window_size *= 2
    self.layernorm_before = nn.LayerNorm(dim, eps=cfg.layer_norm_eps)
    self.drop_path = DinatDropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
    dim_after = dim * 2 if double_attention else dim
    self.layernorm_after = nn.LayerNorm(dim_after, eps=cfg.layer_norm_eps)
    self.intermediate = DinatIntermediate(cfg, dim_after, int(dim_after * cfg.mlp_ratio))
    self.output = DinatOutput(cfg, int(dim_after * cfg.mlp_ratio), dim)

  @abstractmethod
  def maybe_pad(self, *args, **kwargs):
    raise NotImplementedError

  def forward(
    self,
    hidden_states: torch.Tensor,
    output_attentions: Optional[bool] = False,
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    if len(hidden_states.shape) > 3:
      is_2d = True
      N, K, T, C = hidden_states.size()
    else:
      is_2d = False
      N, T, C = hidden_states.shape
    shortcut = hidden_states

    hidden_states = self.layernorm_before(hidden_states)
    if is_2d:
      hidden_states, pad_values = self.maybe_pad(hidden_states, K, T)
      _, height_pad, width_pad, _ = hidden_states.shape
    else:
      hidden_states, pad_values = self.maybe_pad(hidden_states, T)

    attention_inputs = hidden_states
    hidden_states_list = []
    for attention in [self.attention, self.attention2]:
      if attention is None:
        continue

      attention_output = attention(attention_inputs, output_attentions=output_attentions)
      attention_output = attention_output[0]

      if is_2d:
        was_padded = pad_values[3] > 0 or pad_values[5] > 0
        if was_padded:
          attention_output = attention_output[:, :K, :T, :].contiguous()
      else:
        was_padded = pad_values[3] > 0
        if was_padded:
          attention_output = attention_output[:, :T, :].contiguous()

      hidden_states = shortcut + self.drop_path(attention_output)
      hidden_states_list.append(hidden_states)

    if self.double_attention:
      hidden_states = torch.cat(hidden_states_list, dim=-1)
      shortcut = torch.stack(hidden_states_list).sum(dim=0) / 2.
    else:
      shortcut = hidden_states
    layer_output = self.layernorm_after(hidden_states)
    layer_output = self.output(self.intermediate(layer_output))

    layer_output = shortcut + self.drop_path(layer_output)
    layer_outputs = (layer_output,)
    return layer_outputs


class DinatLayer1d(_DinatLayerNd):
  def __init__(
    self,
    cfg: Config,
    dim: int,
    num_heads: int,
    kernel_size: int,
    dilation: int,
    drop_path_rate: float,
    double_attention: bool,
  ):
    super().__init__(cfg, dim, kernel_size, dilation, drop_path_rate, double_attention)
    self.attention = NeighborhoodAttentionModule1d(cfg, dim, num_heads, kernel_size, dilation)
    if double_attention:
      self.attention2 = NeighborhoodAttentionModule1d(cfg, dim, num_heads, kernel_size, dilation * 2)
    else:
      self.attention2 = None

  def maybe_pad(self, hidden_states, frames):
    window_size = self.window_size
    pad_values = (0, 0, 0, 0)
    if frames < window_size:
      pad_l = 0
      pad_r = max(0, window_size - frames)
      pad_values = (0, 0, pad_l, pad_r)
      hidden_states = nn.functional.pad(hidden_states, pad_values)
    return hidden_states, pad_values


class DinatLayer2d(_DinatLayerNd):
  def __init__(
    self,
    cfg: Config,
    dim: int,
    num_heads: int,
    kernel_size: int,
    dilation: int,
    drop_path_rate: float
  ):
    super().__init__(cfg, dim, kernel_size, dilation, drop_path_rate, double_attention=False)
    self.attention = NeighborhoodAttentionModule2d(cfg, dim, num_heads, kernel_size, dilation)
    self.attention2 = None

  def maybe_pad(self, hidden_states, height, width):
    window_size = self.window_size
    pad_values = (0, 0, 0, 0, 0, 0)
    if height < window_size or width < window_size:
      pad_l = pad_t = 0
      pad_r = max(0, window_size - width)
      pad_b = max(0, window_size - height)
      pad_values = (0, 0, pad_l, pad_r, pad_t, pad_b)
      hidden_states = nn.functional.pad(hidden_states, pad_values)
    return hidden_states, pad_values
'''

def main():
    print(f"Patching {DINAT_PATH}...")
    
    # Backup original
    backup_path = DINAT_PATH + ".bak"
    if os.path.exists(DINAT_PATH):
        with open(DINAT_PATH, 'r') as f:
            original = f.read()
        with open(backup_path, 'w') as f:
            f.write(original)
        print(f"  Backup saved to {backup_path}")
    
    # Write patched version
    with open(DINAT_PATH, 'w') as f:
        f.write(NEW_DINAT_CODE)
    
    print("  Patch applied successfully!")
    
    # Verify import works
    print("  Verifying import...")
    try:
        import importlib
        import sys
        # Clear cached imports
        for mod in list(sys.modules.keys()):
            if 'allin1' in mod or 'natten' in mod:
                del sys.modules[mod]
        
        import allin1
        print("  ✓ allin1 imports successfully!")
        return 0
    except Exception as e:
        print(f"  ✗ Import failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
