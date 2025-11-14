"""
Patch encoding using CNN and Vision Transformer backbones.
Extracts feature representations from image patches.
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from typing import Optional, Tuple, Union
import timm
from transformers import ViTModel, ViTConfig


class PatchEncoder(nn.Module):
    """
    Patch encoding module supporting both CNN and ViT backbones.
    """
    
    def __init__(
        self,
        backbone_type: str = 'resnet50',
        pretrained: bool = True,
        feature_dim: int = 512,
        freeze_backbone: bool = False,
        patch_size: int = 256
    ):
        """
        Initialize patch encoder.
        
        Args:
            backbone_type: Type of backbone ('resnet50', 'vit_base', 'convnext', etc.)
            pretrained: Whether to use pretrained weights
            feature_dim: Dimension of output features
            freeze_backbone: Whether to freeze backbone parameters
            patch_size: Input patch size
        """
        super().__init__()
        
        self.backbone_type = backbone_type
        self.feature_dim = feature_dim
        self.patch_size = patch_size
        
        # Initialize backbone
        if backbone_type.startswith('vit'):
            self.backbone = self._init_vit_backbone(backbone_type, pretrained)
            self.is_vit = True
        else:
            self.backbone = self._init_cnn_backbone(backbone_type, pretrained)
            self.is_vit = False
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Get backbone output dimension
        backbone_dim = self._get_backbone_dim()
        
        # Feature projection layer
        self.feature_projection = nn.Sequential(
            nn.Linear(backbone_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Input transforms
        self.transforms = self._get_transforms()
    
    def _init_cnn_backbone(self, backbone_type: str, pretrained: bool) -> nn.Module:
        """Initialize CNN backbone using timm."""
        backbone = timm.create_model(
            backbone_type,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool=''  # Remove global pooling
        )
        return backbone
    
    def _init_vit_backbone(self, backbone_type: str, pretrained: bool) -> nn.Module:
        """Initialize Vision Transformer backbone."""
        if pretrained:
            if backbone_type == 'vit_base':
                model_name = 'google/vit-base-patch16-224'
            elif backbone_type == 'vit_large':
                model_name = 'google/vit-large-patch16-224'
            else:
                model_name = 'google/vit-base-patch16-224'
            
            backbone = ViTModel.from_pretrained(model_name)
        else:
            config = ViTConfig()
            backbone = ViTModel(config)
        
        return backbone
    
    def _get_backbone_dim(self) -> int:
        """Get the output dimension of the backbone."""
        if self.is_vit:
            return self.backbone.config.hidden_size
        else:
            # Test with dummy input to get output dimension
            dummy_input = torch.randn(1, 3, self.patch_size, self.patch_size)
            with torch.no_grad():
                output = self.backbone(dummy_input)
                if isinstance(output, tuple):
                    output = output[0]
                
                # Handle different output shapes
                if len(output.shape) == 4:  # [B, C, H, W]
                    return output.shape[1]
                elif len(output.shape) == 3:  # [B, L, C]
                    return output.shape[-1]
                else:  # [B, C]
                    return output.shape[-1]
    
    def _get_transforms(self) -> transforms.Compose:
        """Get input transforms for the backbone."""
        if self.is_vit:
            # ViT typically expects 224x224 inputs
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            # CNN backbones can handle variable sizes
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((self.patch_size, self.patch_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
    
    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through patch encoder.
        
        Args:
            patches: Batch of patches [B, C, H, W] or [B, N, C, H, W]
            
        Returns:
            Encoded features [B, feature_dim] or [B, N, feature_dim]
        """
        original_shape = patches.shape
        
        # Handle both single batch and multi-patch inputs
        if len(patches.shape) == 5:  # [B, N, C, H, W]
            batch_size, num_patches = patches.shape[:2]
            patches = patches.view(-1, *patches.shape[2:])  # [B*N, C, H, W]
            reshape_needed = True
        else:
            reshape_needed = False
        
        # Extract features using backbone
        if self.is_vit:
            outputs = self.backbone(patches)
            features = outputs.last_hidden_state[:, 0]  # Use CLS token
        else:
            features = self.backbone(patches)
            
            # Handle different CNN output formats
            if len(features.shape) == 4:  # [B, C, H, W]
                features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
                features = features.view(features.size(0), -1)
            elif len(features.shape) == 3:  # [B, L, C]
                features = features.mean(dim=1)  # Global average pooling
        
        # Project to desired feature dimension
        features = self.feature_projection(features)
        
        # Reshape back if needed
        if reshape_needed:
            features = features.view(batch_size, num_patches, self.feature_dim)
        
        return features


class MultiScalePatchEncoder(nn.Module):
    """
    Multi-scale patch encoder that processes patches at different resolutions.
    """
    
    def __init__(
        self,
        backbone_type: str = 'resnet50',
        scales: Tuple[int, ...] = (224, 256, 384),
        pretrained: bool = True,
        feature_dim: int = 512,
        fusion_method: str = 'concat'
    ):
        """
        Initialize multi-scale patch encoder.
        
        Args:
            backbone_type: Type of backbone
            scales: Different scales to process patches
            pretrained: Whether to use pretrained weights
            feature_dim: Output feature dimension
            fusion_method: How to fuse multi-scale features ('concat', 'sum', 'attention')
        """
        super().__init__()
        
        self.scales = scales
        self.fusion_method = fusion_method
        
        # Create encoder for each scale
        self.encoders = nn.ModuleList([
            PatchEncoder(
                backbone_type=backbone_type,
                pretrained=pretrained,
                feature_dim=feature_dim,
                patch_size=scale
            )
            for scale in scales
        ])
        
        # Fusion layer
        if fusion_method == 'concat':
            self.fusion = nn.Linear(len(scales) * feature_dim, feature_dim)
        elif fusion_method == 'attention':
            self.attention = nn.MultiheadAttention(feature_dim, num_heads=8)
            self.fusion = nn.Linear(feature_dim, feature_dim)
        else:  # sum
            self.fusion = nn.Identity()
    
    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through multi-scale encoder.
        
        Args:
            patches: Input patches [B, C, H, W]
            
        Returns:
            Fused multi-scale features [B, feature_dim]
        """
        scale_features = []
        
        for encoder in self.encoders:
            # Resize patches to match encoder's expected size
            resized_patches = torch.nn.functional.interpolate(
                patches, 
                size=(encoder.patch_size, encoder.patch_size),
                mode='bilinear',
                align_corners=False
            )
            features = encoder(resized_patches)
            scale_features.append(features)
        
        # Fuse features from different scales
        if self.fusion_method == 'concat':
            fused_features = torch.cat(scale_features, dim=-1)
            fused_features = self.fusion(fused_features)
        elif self.fusion_method == 'attention':
            # Stack features for attention
            stacked_features = torch.stack(scale_features, dim=1)  # [B, num_scales, feature_dim]
            attended_features, _ = self.attention(
                stacked_features, stacked_features, stacked_features
            )
            fused_features = attended_features.mean(dim=1)
            fused_features = self.fusion(fused_features)
        else:  # sum
            fused_features = torch.stack(scale_features, dim=0).sum(dim=0)
        
        return fused_features


def create_patch_encoder(
    backbone_type: str = 'resnet50',
    pretrained: bool = True,
    feature_dim: int = 512,
    multi_scale: bool = False,
    **kwargs
) -> Union[PatchEncoder, MultiScalePatchEncoder]:
    """
    Factory function to create patch encoder.
    
    Args:
        backbone_type: Type of backbone
        pretrained: Whether to use pretrained weights
        feature_dim: Output feature dimension
        multi_scale: Whether to use multi-scale encoding
        **kwargs: Additional arguments
        
    Returns:
        Patch encoder instance
    """
    if multi_scale:
        return MultiScalePatchEncoder(
            backbone_type=backbone_type,
            pretrained=pretrained,
            feature_dim=feature_dim,
            **kwargs
        )
    else:
        return PatchEncoder(
            backbone_type=backbone_type,
            pretrained=pretrained,
            feature_dim=feature_dim,
            **kwargs
        )