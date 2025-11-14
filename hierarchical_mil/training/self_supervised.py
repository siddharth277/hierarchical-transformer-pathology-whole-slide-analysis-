"""
Self-supervised learning components for hierarchical MIL.
Includes SimCLR, MoCo-style contrastive learning, and masked patch prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import random
import numpy as np


class SimCLRProjectionHead(nn.Module):
    """
    Projection head for SimCLR-style contrastive learning.
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 512,
        output_dim: int = 128
    ):
        """
        Initialize projection head.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output projection dimension
        """
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss (NT-Xent) for contrastive learning.
    Used in SimCLR.
    """

    def __init__(self, temperature: float = 0.5):
        """
        Initialize NT-Xent loss.

        Args:
            temperature: Temperature parameter for scaling
        """
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        Compute NT-Xent loss.

        Args:
            z_i: Embeddings from augmentation 1 [B, dim]
            z_j: Embeddings from augmentation 2 [B, dim]

        Returns:
            Contrastive loss
        """
        batch_size = z_i.size(0)

        # Normalize embeddings
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        # Concatenate embeddings
        representations = torch.cat([z_i, z_j], dim=0)  # [2B, dim]

        # Compute similarity matrix
        similarity_matrix = torch.matmul(representations, representations.T)  # [2B, 2B]
        similarity_matrix = similarity_matrix / self.temperature

        # Create positive pair mask
        positives_mask = torch.zeros(2 * batch_size, 2 * batch_size, dtype=torch.bool, device=z_i.device)
        positives_mask[:batch_size, batch_size:] = torch.eye(batch_size, dtype=torch.bool, device=z_i.device)
        positives_mask[batch_size:, :batch_size] = torch.eye(batch_size, dtype=torch.bool, device=z_i.device)

        # Create negative mask (all except self and positive pair)
        negatives_mask = ~torch.eye(2 * batch_size, dtype=torch.bool, device=z_i.device)

        # Compute log probability
        exp_sim = torch.exp(similarity_matrix) * negatives_mask
        log_prob = similarity_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))

        # Mean log-likelihood over positive pairs
        mean_log_prob_pos = (positives_mask * log_prob).sum(dim=1) / positives_mask.sum(dim=1)
        loss = -mean_log_prob_pos.mean()

        return loss


class MaskedPatchPrediction(nn.Module):
    """
    Masked patch prediction for self-supervised pre-training.
    Similar to MAE (Masked Autoencoders).
    """

    def __init__(
        self,
        patch_encoder: nn.Module,
        feature_dim: int = 512,
        mask_ratio: float = 0.75,
        decoder_dim: int = 256,
        decoder_depth: int = 2
    ):
        """
        Initialize masked patch prediction.

        Args:
            patch_encoder: Patch encoder module
            feature_dim: Feature dimension
            mask_ratio: Ratio of patches to mask
            decoder_dim: Decoder hidden dimension
            decoder_depth: Number of decoder layers
        """
        super().__init__()
        self.patch_encoder = patch_encoder
        self.mask_ratio = mask_ratio

        # Mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, feature_dim))

        # Lightweight decoder
        decoder_layers = []
        for i in range(decoder_depth):
            decoder_layers.extend([
                nn.Linear(feature_dim if i == 0 else decoder_dim, decoder_dim),
                nn.ReLU(inplace=True)
            ])
        decoder_layers.append(nn.Linear(decoder_dim, feature_dim))

        self.decoder = nn.Sequential(*decoder_layers)

        # Initialize mask token
        torch.nn.init.normal_(self.mask_token, std=0.02)

    def random_masking(
        self,
        features: torch.Tensor,
        mask_ratio: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform random masking.

        Args:
            features: Input features [B, N, dim]
            mask_ratio: Masking ratio (uses self.mask_ratio if None)

        Returns:
            Masked features, mask, and restore indices
        """
        B, N, D = features.shape
        mask_ratio = mask_ratio or self.mask_ratio

        # Calculate number of patches to keep
        num_keep = int(N * (1 - mask_ratio))

        # Random shuffle
        noise = torch.rand(B, N, device=features.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Keep the first subset
        ids_keep = ids_shuffle[:, :num_keep]

        # Generate mask: 0 is keep, 1 is remove
        mask = torch.ones(B, N, device=features.device)
        mask[:, :num_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        # Get masked features
        features_masked = torch.gather(
            features,
            dim=1,
            index=ids_keep.unsqueeze(-1).expand(-1, -1, D)
        )

        return features_masked, mask, ids_restore

    def forward(
        self,
        patches: torch.Tensor,
        return_loss: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with masked patch prediction.

        Args:
            patches: Input patches [B, N, C, H, W]
            return_loss: Whether to compute reconstruction loss

        Returns:
            Dictionary with loss and predictions
        """
        B, N = patches.shape[:2]

        # Encode all patches
        with torch.no_grad():
            patch_features = self.patch_encoder(patches.view(-1, *patches.shape[2:]))
            patch_features = patch_features.view(B, N, -1)

        # Random masking
        masked_features, mask, ids_restore = self.random_masking(patch_features)

        # Add mask tokens
        mask_tokens = self.mask_token.repeat(B, N - masked_features.size(1), 1)
        full_features = torch.cat([masked_features, mask_tokens], dim=1)

        # Unshuffle
        full_features = torch.gather(
            full_features,
            dim=1,
            index=ids_restore.unsqueeze(-1).expand(-1, -1, full_features.shape[-1])
        )

        # Decoder
        predictions = self.decoder(full_features)

        output = {'predictions': predictions, 'mask': mask}

        if return_loss:
            # Reconstruction loss (only on masked patches)
            loss = F.mse_loss(predictions, patch_features.detach(), reduction='none')
            loss = (loss.mean(dim=-1) * mask).sum() / mask.sum()
            output['loss'] = loss

        return output


class InstanceContrastiveLearning(nn.Module):
    """
    Instance-level contrastive learning for MIL.
    Learns to contrast positive instances vs negative instances within a slide.
    """

    def __init__(
        self,
        feature_dim: int = 512,
        projection_dim: int = 128,
        temperature: float = 0.5,
        queue_size: int = 4096
    ):
        """
        Initialize instance contrastive learning.

        Args:
            feature_dim: Input feature dimension
            projection_dim: Projection dimension
            temperature: Temperature for contrastive loss
            queue_size: Size of negative queue
        """
        super().__init__()
        self.temperature = temperature
        self.queue_size = queue_size

        # Projection head
        self.projection_head = SimCLRProjectionHead(
            input_dim=feature_dim,
            hidden_dim=feature_dim,
            output_dim=projection_dim
        )

        # Queue for negative samples
        self.register_buffer("queue", torch.randn(projection_dim, queue_size))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor):
        """Update the queue with new keys."""
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)

        # Replace oldest entries
        if ptr + batch_size <= self.queue_size:
            self.queue[:, ptr:ptr + batch_size] = keys.T
        else:
            # Wrap around
            remaining = self.queue_size - ptr
            self.queue[:, ptr:] = keys[:remaining].T
            self.queue[:, :batch_size - remaining] = keys[remaining:].T

        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr

    def forward(
        self,
        instance_features: torch.Tensor,
        positive_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute instance contrastive loss.

        Args:
            instance_features: Instance features [B, N, dim]
            positive_mask: Mask indicating positive instances [B, N]

        Returns:
            Contrastive loss
        """
        B, N, D = instance_features.shape

        # Project features
        features_flat = instance_features.view(-1, D)
        projections = self.projection_head(features_flat)  # [B*N, projection_dim]
        projections = F.normalize(projections, dim=1)

        # Separate positive and negative instances
        positive_mask_flat = positive_mask.view(-1)
        positive_features = projections[positive_mask_flat]
        negative_features = projections[~positive_mask_flat]

        if len(positive_features) == 0 or len(negative_features) == 0:
            return torch.tensor(0.0, device=instance_features.device)

        # Compute similarities
        pos_sim = torch.mm(positive_features, positive_features.T) / self.temperature
        neg_sim = torch.mm(positive_features, self.queue.clone().detach()) / self.temperature

        # Compute contrastive loss
        logits = torch.cat([pos_sim, neg_sim], dim=1)
        labels = torch.arange(len(positive_features), device=instance_features.device)

        loss = F.cross_entropy(logits, labels)

        # Update queue
        self._dequeue_and_enqueue(negative_features.detach())

        return loss


class SelfSupervisedPretrainer:
    """
    Self-supervised pretraining wrapper for hierarchical MIL models.
    """

    def __init__(
        self,
        model: nn.Module,
        method: str = 'simclr',
        temperature: float = 0.5,
        projection_dim: int = 128
    ):
        """
        Initialize self-supervised pretrainer.

        Args:
            model: Base model to pretrain
            method: Pretraining method ('simclr', 'masked_prediction', 'instance_contrastive')
            temperature: Temperature for contrastive loss
            projection_dim: Projection dimension
        """
        self.model = model
        self.method = method

        if method == 'simclr':
            self.projection_head = SimCLRProjectionHead(
                input_dim=model.slide_dim,
                output_dim=projection_dim
            )
            self.criterion = NTXentLoss(temperature=temperature)

        elif method == 'masked_prediction':
            self.masked_predictor = MaskedPatchPrediction(
                patch_encoder=model.patch_encoder,
                feature_dim=model.patch_feature_dim
            )

        elif method == 'instance_contrastive':
            self.instance_contrastive = InstanceContrastiveLearning(
                feature_dim=model.region_dim,
                projection_dim=projection_dim,
                temperature=temperature
            )

    def pretrain_step(
        self,
        batch: Dict[str, torch.Tensor],
        augment_fn: Optional[callable] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Perform one pretraining step.

        Args:
            batch: Input batch
            augment_fn: Augmentation function for creating views

        Returns:
            Loss and metrics dict
        """
        if self.method == 'simclr':
            # Create two augmented views
            view1 = augment_fn(batch['patches']) if augment_fn else batch['patches']
            view2 = augment_fn(batch['patches']) if augment_fn else batch['patches']

            # Forward pass
            out1 = self.model(view1, return_embeddings=True)
            out2 = self.model(view2, return_embeddings=True)

            # Project embeddings
            z1 = self.projection_head(out1['slide_embeddings'])
            z2 = self.projection_head(out2['slide_embeddings'])

            # Compute loss
            loss = self.criterion(z1, z2)
            metrics = {'contrastive_loss': loss.item()}

        elif self.method == 'masked_prediction':
            # Masked patch prediction
            output = self.masked_predictor(batch['patches'], return_loss=True)
            loss = output['loss']
            metrics = {'reconstruction_loss': loss.item()}

        elif self.method == 'instance_contrastive':
            # Forward pass to get instance features
            outputs = self.model(batch['patches'], return_embeddings=True)
            instance_features = outputs['region_embeddings']

            # Assume positive instances based on attention (top-k)
            if 'region_attention' in outputs:
                attn = outputs['region_attention']
                k = max(1, int(instance_features.size(1) * 0.3))
                _, top_idx = torch.topk(attn.mean(dim=1), k, dim=1)
                positive_mask = torch.zeros_like(attn, dtype=torch.bool)
                positive_mask.scatter_(1, top_idx, True)
            else:
                # Random selection
                positive_mask = torch.rand_like(instance_features[:, :, 0]) > 0.7

            loss = self.instance_contrastive(instance_features, positive_mask)
            metrics = {'instance_contrastive_loss': loss.item()}

        return loss, metrics
