"""
Attention-based aggregation modules for hierarchical MIL.
Implements patch-to-region and region-to-slide aggregation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from einops import rearrange, repeat


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention module with optional positional encoding.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_dropout: float = 0.1,
        proj_dropout: float = 0.1,
        use_positional_encoding: bool = False,
        max_seq_len: int = 1000
    ):
        super().__init__()
        
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_dropout = nn.Dropout(proj_dropout)
        
        # Optional positional encoding
        self.use_positional_encoding = use_positional_encoding
        if use_positional_encoding:
            self.positional_encoding = PositionalEncoding(dim, max_seq_len)
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, N, dim]
            mask: Attention mask [B, N] (1 for valid, 0 for masked)
            return_attention: Whether to return attention weights
            
        Returns:
            Tuple of (output, attention_weights)
        """
        B, N, C = x.shape
        
        # Apply positional encoding if enabled
        if self.use_positional_encoding:
            x = self.positional_encoding(x)
        
        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, head_dim]
        q, k, v = qkv.unbind(0)
        
        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, N]
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_dropout(x)
        
        if return_attention:
            return x, attn.mean(dim=1)  # Average over heads
        else:
            return x, None


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for sequences.
    """
    
    def __init__(self, dim: int, max_seq_len: int = 1000):
        super().__init__()
        
        pe = torch.zeros(max_seq_len, dim)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * -(math.log(10000.0) / dim)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_seq_len, dim]
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor [B, N, dim]
            
        Returns:
            Input with added positional encoding
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class AttentionAggregator(nn.Module):
    """
    Attention-based aggregation module for MIL.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        aggregation_method: str = 'attention_pooling'
    ):
        """
        Initialize attention aggregator.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout rate
            aggregation_method: 'attention_pooling', 'cls_token', or 'mean_pooling'
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.aggregation_method = aggregation_method
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(
                dim=hidden_dim,
                num_heads=num_heads,
                mlp_ratio=4.0,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Aggregation components
        if aggregation_method == 'attention_pooling':
            self.attention_pool = nn.MultiheadAttention(
                hidden_dim, num_heads=1, batch_first=True
            )
            self.query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        elif aggregation_method == 'cls_token':
            self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Input features [B, N, input_dim]
            mask: Mask for valid features [B, N]
            return_attention: Whether to return attention weights
            
        Returns:
            Tuple of (aggregated_features, attention_weights)
        """
        B, N, _ = x.shape
        
        # Project input
        x = self.input_proj(x)
        
        # Add CLS token if using cls_token aggregation
        if self.aggregation_method == 'cls_token':
            cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=B)
            x = torch.cat([cls_tokens, x], dim=1)
            if mask is not None:
                # Add mask for CLS token (always valid)
                cls_mask = torch.ones(B, 1, device=mask.device, dtype=mask.dtype)
                mask = torch.cat([cls_mask, mask], dim=1)
        
        # Apply transformer layers
        attention_weights = None
        for i, layer in enumerate(self.layers):
            x, attn = layer(x, mask, return_attention=(i == len(self.layers) - 1 and return_attention))
            if attn is not None:
                attention_weights = attn
        
        # Aggregate features
        if self.aggregation_method == 'cls_token':
            output = x[:, 0]  # Use CLS token
        elif self.aggregation_method == 'attention_pooling':
            query = repeat(self.query, '1 1 d -> b 1 d', b=B)
            output, pool_attn = self.attention_pool(query, x, x, key_padding_mask=(mask == 0) if mask is not None else None)
            output = output.squeeze(1)
            if return_attention and attention_weights is None:
                attention_weights = pool_attn
        else:  # mean_pooling
            if mask is not None:
                mask_expanded = mask.unsqueeze(-1).expand_as(x)
                x_masked = x * mask_expanded
                output = x_masked.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
            else:
                output = x.mean(dim=1)
        
        output = self.norm(output)
        
        return output, attention_weights


class TransformerLayer(nn.Module):
    """
    Transformer layer with multi-head attention and MLP.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        drop_path: float = 0.0
    ):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(
            dim=dim,
            num_heads=num_heads,
            attn_dropout=dropout,
            proj_dropout=dropout
        )
        
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
        # Drop path (stochastic depth)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, N, dim]
            mask: Attention mask [B, N]
            return_attention: Whether to return attention weights
            
        Returns:
            Tuple of (output, attention_weights)
        """
        # Multi-head attention
        attn_output, attention_weights = self.attn(
            self.norm1(x), mask, return_attention
        )
        x = x + self.drop_path(attn_output)
        
        # MLP
        mlp_output = self.mlp(self.norm2(x))
        x = x + self.drop_path(mlp_output)
        
        return x, attention_weights


class DropPath(nn.Module):
    """
    Drop paths (stochastic depth) per sample.
    """
    
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class PatchToRegionAggregator(AttentionAggregator):
    """
    Aggregates patches into region representations using attention.
    """
    
    def __init__(
        self,
        patch_dim: int,
        region_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_patches_per_region: int = 100
    ):
        super().__init__(
            input_dim=patch_dim,
            hidden_dim=region_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            aggregation_method='attention_pooling'
        )
        
        self.max_patches_per_region = max_patches_per_region


class RegionToSlideAggregator(AttentionAggregator):
    """
    Aggregates region representations into slide-level representation using Transformer.
    """
    
    def __init__(
        self,
        region_dim: int,
        slide_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        max_regions_per_slide: int = 50,
        use_positional_encoding: bool = True
    ):
        super().__init__(
            input_dim=region_dim,
            hidden_dim=slide_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            aggregation_method='cls_token'
        )
        
        self.max_regions_per_slide = max_regions_per_slide
        
        # Add positional encoding to first layer if requested
        if use_positional_encoding and len(self.layers) > 0:
            self.layers[0].attn.use_positional_encoding = True
            self.layers[0].attn.positional_encoding = PositionalEncoding(
                slide_dim, max_regions_per_slide + 1  # +1 for CLS token
            )