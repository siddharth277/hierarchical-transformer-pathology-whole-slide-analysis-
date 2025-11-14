"""
Hierarchical MIL Transformer model.
Implements the complete pipeline: patches → regions → slide representation → classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from .patch_encoder import PatchEncoder
from .attention import PatchToRegionAggregator, RegionToSlideAggregator


class HierarchicalMILTransformer(nn.Module):
    """
    Hierarchical MIL Transformer for whole slide image analysis.
    
    Architecture:
    1. Patch Encoding: CNN/ViT backbone encodes image patches
    2. Patch-to-Region: Attention aggregates patches into region embeddings
    3. Region-to-Slide: Transformer aggregates regions into slide representation
    4. Classification: MIL classifier predicts slide-level labels
    """
    
    def __init__(
        self,
        # Patch encoder config
        backbone_type: str = 'resnet50',
        patch_feature_dim: int = 512,
        pretrained_backbone: bool = True,
        freeze_backbone: bool = False,
        
        # Region aggregation config
        region_dim: int = 512,
        patches_per_region: int = 64,
        region_attention_heads: int = 8,
        region_attention_layers: int = 2,
        
        # Slide aggregation config
        slide_dim: int = 512,
        regions_per_slide: int = 32,
        slide_attention_heads: int = 8,
        slide_attention_layers: int = 4,
        
        # Classification config
        num_classes: int = 2,
        dropout: float = 0.1,
        
        # Training config
        use_gradient_checkpointing: bool = False
    ):
        """
        Initialize Hierarchical MIL Transformer.
        
        Args:
            backbone_type: Type of patch encoder backbone
            patch_feature_dim: Dimension of patch features
            pretrained_backbone: Whether to use pretrained backbone
            freeze_backbone: Whether to freeze backbone parameters
            region_dim: Dimension of region representations
            patches_per_region: Maximum patches per region
            region_attention_heads: Number of attention heads for region aggregation
            region_attention_layers: Number of layers for region aggregation
            slide_dim: Dimension of slide representation
            regions_per_slide: Maximum regions per slide
            slide_attention_heads: Number of attention heads for slide aggregation
            slide_attention_layers: Number of layers for slide aggregation
            num_classes: Number of output classes
            dropout: Dropout rate
            use_gradient_checkpointing: Whether to use gradient checkpointing
        """
        super().__init__()
        
        self.patch_feature_dim = patch_feature_dim
        self.region_dim = region_dim
        self.slide_dim = slide_dim
        self.patches_per_region = patches_per_region
        self.regions_per_slide = regions_per_slide
        self.num_classes = num_classes
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # 1. Patch Encoder
        self.patch_encoder = PatchEncoder(
            backbone_type=backbone_type,
            feature_dim=patch_feature_dim,
            pretrained=pretrained_backbone,
            freeze_backbone=freeze_backbone
        )
        
        # 2. Patch-to-Region Aggregator
        self.patch_to_region = PatchToRegionAggregator(
            patch_dim=patch_feature_dim,
            region_dim=region_dim,
            num_heads=region_attention_heads,
            num_layers=region_attention_layers,
            dropout=dropout,
            max_patches_per_region=patches_per_region
        )
        
        # 3. Region-to-Slide Aggregator
        self.region_to_slide = RegionToSlideAggregator(
            region_dim=region_dim,
            slide_dim=slide_dim,
            num_heads=slide_attention_heads,
            num_layers=slide_attention_layers,
            dropout=dropout,
            max_regions_per_slide=regions_per_slide,
            use_positional_encoding=True
        )
        
        # 4. MIL Classifier
        self.classifier = MILClassifier(
            input_dim=slide_dim,
            num_classes=num_classes,
            dropout=dropout
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(
        self,
        patches: torch.Tensor,
        region_masks: Optional[torch.Tensor] = None,
        slide_masks: Optional[torch.Tensor] = None,
        return_attention: bool = False,
        return_embeddings: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through hierarchical MIL transformer.
        
        Args:
            patches: Input patches [B, R, P, C, H, W] where:
                     B = batch size, R = regions per slide, P = patches per region
            region_masks: Mask for valid patches [B, R, P]
            slide_masks: Mask for valid regions [B, R]
            return_attention: Whether to return attention weights
            return_embeddings: Whether to return intermediate embeddings
            
        Returns:
            Dictionary containing:
            - logits: Classification logits [B, num_classes]
            - probabilities: Class probabilities [B, num_classes]
            - slide_embeddings: Slide-level embeddings [B, slide_dim] (if return_embeddings)
            - region_embeddings: Region-level embeddings [B, R, region_dim] (if return_embeddings)
            - patch_embeddings: Patch-level embeddings [B, R, P, patch_feature_dim] (if return_embeddings)
            - region_attention: Region attention weights (if return_attention)
            - slide_attention: Slide attention weights (if return_attention)
        """
        batch_size, num_regions, patches_per_region = patches.shape[:3]
        
        # Reshape patches for batch processing
        patches_flat = patches.view(-1, *patches.shape[3:])  # [B*R*P, C, H, W]
        
        # 1. Encode patches
        if self.use_gradient_checkpointing and self.training:
            patch_features = torch.utils.checkpoint.checkpoint(
                self.patch_encoder, patches_flat
            )
        else:
            patch_features = self.patch_encoder(patches_flat)
        
        # Reshape back to hierarchical structure
        patch_features = patch_features.view(
            batch_size, num_regions, patches_per_region, self.patch_feature_dim
        )  # [B, R, P, patch_feature_dim]
        
        # 2. Aggregate patches to regions
        region_features = []
        region_attentions = [] if return_attention else None
        
        for i in range(num_regions):
            # Get patches for this region
            region_patches = patch_features[:, i]  # [B, P, patch_feature_dim]
            region_mask = region_masks[:, i] if region_masks is not None else None
            
            # Aggregate patches to region
            if self.use_gradient_checkpointing and self.training:
                region_feat, region_att = torch.utils.checkpoint.checkpoint(
                    self.patch_to_region, region_patches, region_mask, return_attention
                )
            else:
                region_feat, region_att = self.patch_to_region(
                    region_patches, region_mask, return_attention
                )
            
            region_features.append(region_feat)
            if return_attention and region_att is not None:
                region_attentions.append(region_att)
        
        # Stack region features
        region_features = torch.stack(region_features, dim=1)  # [B, R, region_dim]
        
        # 3. Aggregate regions to slide
        if self.use_gradient_checkpointing and self.training:
            slide_features, slide_attention = torch.utils.checkpoint.checkpoint(
                self.region_to_slide, region_features, slide_masks, return_attention
            )
        else:
            slide_features, slide_attention = self.region_to_slide(
                region_features, slide_masks, return_attention
            )
        
        # 4. Classification
        logits, probabilities = self.classifier(slide_features)
        
        # Prepare output
        output = {
            'logits': logits,
            'probabilities': probabilities
        }
        
        if return_embeddings:
            output.update({
                'slide_embeddings': slide_features,
                'region_embeddings': region_features,
                'patch_embeddings': patch_features
            })
        
        if return_attention:
            if region_attentions:
                output['region_attention'] = torch.stack(region_attentions, dim=1)
            if slide_attention is not None:
                output['slide_attention'] = slide_attention
        
        return output
    
    def encode_slide(
        self,
        patches: torch.Tensor,
        region_masks: Optional[torch.Tensor] = None,
        slide_masks: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode slide to get slide-level representation.
        
        Args:
            patches: Input patches
            region_masks: Region masks
            slide_masks: Slide masks
            
        Returns:
            Slide-level embeddings [B, slide_dim]
        """
        with torch.no_grad():
            output = self.forward(
                patches, region_masks, slide_masks, 
                return_attention=False, return_embeddings=True
            )
            return output['slide_embeddings']


class MILClassifier(nn.Module):
    """
    MIL classifier for slide-level prediction.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        use_attention: bool = True
    ):
        """
        Initialize MIL classifier.
        
        Args:
            input_dim: Input feature dimension
            num_classes: Number of output classes
            dropout: Dropout rate
            use_attention: Whether to use attention-based MIL
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.use_attention = use_attention
        
        if use_attention:
            # Attention-based MIL
            self.attention = nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.Tanh(),
                nn.Linear(input_dim // 2, 1)
            )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, num_classes)
        )
    
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through MIL classifier.
        
        Args:
            features: Input features [B, input_dim] or [B, N, input_dim]
            
        Returns:
            Tuple of (logits, probabilities)
        """
        if len(features.shape) == 3 and self.use_attention:
            # Multiple instance learning with attention
            B, N, D = features.shape
            
            # Compute attention weights
            attention_weights = self.attention(features)  # [B, N, 1]
            attention_weights = F.softmax(attention_weights, dim=1)
            
            # Aggregate features using attention
            features = (features * attention_weights).sum(dim=1)  # [B, D]
        elif len(features.shape) == 3:
            # Simple average pooling
            features = features.mean(dim=1)
        
        # Classification
        logits = self.classifier(features)
        probabilities = F.softmax(logits, dim=-1)
        
        return logits, probabilities


def create_hierarchical_mil_transformer(
    config: Dict,
    num_classes: int = 2
) -> HierarchicalMILTransformer:
    """
    Factory function to create hierarchical MIL transformer from config.
    
    Args:
        config: Configuration dictionary
        num_classes: Number of output classes
        
    Returns:
        HierarchicalMILTransformer instance
    """
    return HierarchicalMILTransformer(
        backbone_type=config.get('backbone_type', 'resnet50'),
        patch_feature_dim=config.get('patch_feature_dim', 512),
        pretrained_backbone=config.get('pretrained_backbone', True),
        freeze_backbone=config.get('freeze_backbone', False),
        region_dim=config.get('region_dim', 512),
        patches_per_region=config.get('patches_per_region', 64),
        region_attention_heads=config.get('region_attention_heads', 8),
        region_attention_layers=config.get('region_attention_layers', 2),
        slide_dim=config.get('slide_dim', 512),
        regions_per_slide=config.get('regions_per_slide', 32),
        slide_attention_heads=config.get('slide_attention_heads', 8),
        slide_attention_layers=config.get('slide_attention_layers', 4),
        num_classes=num_classes,
        dropout=config.get('dropout', 0.1),
        use_gradient_checkpointing=config.get('use_gradient_checkpointing', False)
    )