"""
Advanced loss functions for hierarchical MIL.
Includes focal loss, instance-level MIL losses, and survival losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    Reference: Lin et al. "Focal Loss for Dense Object Detection" (ICCV 2017)
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        """
        Initialize Focal Loss.

        Args:
            alpha: Class weights [num_classes]
            gamma: Focusing parameter (gamma > 0 reduces loss for well-classified examples)
            reduction: 'none', 'mean', or 'sum'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            logits: Model predictions [B, num_classes]
            targets: Ground truth labels [B]

        Returns:
            Focal loss value
        """
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # Probability of correct class
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class InstanceMILLoss(nn.Module):
    """
    Instance-level MIL loss that considers individual instance predictions.
    Combines slide-level and instance-level supervision.
    """

    def __init__(
        self,
        slide_loss_weight: float = 1.0,
        instance_loss_weight: float = 0.5,
        max_instances_per_class: int = 10
    ):
        """
        Initialize Instance MIL Loss.

        Args:
            slide_loss_weight: Weight for slide-level loss
            instance_loss_weight: Weight for instance-level loss
            max_instances_per_class: Maximum instances to use for instance loss
        """
        super().__init__()
        self.slide_loss_weight = slide_loss_weight
        self.instance_loss_weight = instance_loss_weight
        self.max_instances_per_class = max_instances_per_class

    def forward(
        self,
        slide_logits: torch.Tensor,
        instance_logits: Optional[torch.Tensor],
        targets: torch.Tensor,
        instance_masks: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute MIL loss.

        Args:
            slide_logits: Slide-level predictions [B, num_classes]
            instance_logits: Instance-level predictions [B, N, num_classes]
            targets: Slide-level labels [B]
            instance_masks: Valid instance mask [B, N]

        Returns:
            Total loss and dict of individual loss components
        """
        # Slide-level loss
        slide_loss = F.cross_entropy(slide_logits, targets)
        total_loss = self.slide_loss_weight * slide_loss

        loss_dict = {'slide_loss': slide_loss.item()}

        # Instance-level loss (if available)
        if instance_logits is not None and self.instance_loss_weight > 0:
            B, N, C = instance_logits.shape

            # Reshape for loss computation
            instance_logits_flat = instance_logits.view(-1, C)
            instance_targets = targets.unsqueeze(1).expand(-1, N).reshape(-1)

            # Apply mask if provided
            if instance_masks is not None:
                valid_instances = instance_masks.view(-1)
                instance_logits_flat = instance_logits_flat[valid_instances]
                instance_targets = instance_targets[valid_instances]

            # Compute instance loss
            if len(instance_logits_flat) > 0:
                instance_loss = F.cross_entropy(instance_logits_flat, instance_targets)
                total_loss += self.instance_loss_weight * instance_loss
                loss_dict['instance_loss'] = instance_loss.item()

        loss_dict['total_loss'] = total_loss.item()
        return total_loss, loss_dict


class MaxPoolingMILLoss(nn.Module):
    """
    Max-pooling MIL loss that focuses on the most confident instances.
    """

    def __init__(self, temperature: float = 1.0):
        """
        Initialize max-pooling MIL loss.

        Args:
            temperature: Temperature for softmax (lower = more focused on max)
        """
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        instance_logits: torch.Tensor,
        targets: torch.Tensor,
        instance_masks: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute max-pooling MIL loss.

        Args:
            instance_logits: Instance predictions [B, N, num_classes]
            targets: Slide labels [B]
            instance_masks: Valid instance mask [B, N]

        Returns:
            Loss value
        """
        B, N, C = instance_logits.shape

        # Apply mask
        if instance_masks is not None:
            instance_logits = instance_logits.clone()
            instance_logits[~instance_masks] = float('-inf')

        # Max pooling over instances
        max_logits, _ = instance_logits.max(dim=1)  # [B, num_classes]

        # Compute loss
        loss = F.cross_entropy(max_logits / self.temperature, targets)
        return loss


class AttentionBasedMILLoss(nn.Module):
    """
    Attention-based MIL loss that uses attention weights for instance selection.
    """

    def __init__(
        self,
        attention_loss_weight: float = 0.1,
        entropy_loss_weight: float = 0.01
    ):
        """
        Initialize attention-based MIL loss.

        Args:
            attention_loss_weight: Weight for attention regularization
            entropy_loss_weight: Weight for attention entropy regularization
        """
        super().__init__()
        self.attention_loss_weight = attention_loss_weight
        self.entropy_loss_weight = entropy_loss_weight

    def forward(
        self,
        slide_logits: torch.Tensor,
        targets: torch.Tensor,
        attention_weights: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute attention-based MIL loss.

        Args:
            slide_logits: Slide predictions [B, num_classes]
            targets: Slide labels [B]
            attention_weights: Attention weights [B, N]

        Returns:
            Total loss and loss components dict
        """
        # Classification loss
        cls_loss = F.cross_entropy(slide_logits, targets)
        total_loss = cls_loss

        loss_dict = {'cls_loss': cls_loss.item()}

        # Attention regularization
        if attention_weights is not None and self.attention_loss_weight > 0:
            # Encourage sparse attention (L1 regularization)
            attention_reg = attention_weights.abs().mean()
            total_loss += self.attention_loss_weight * attention_reg
            loss_dict['attention_reg'] = attention_reg.item()

            # Encourage low entropy (focused attention)
            if self.entropy_loss_weight > 0:
                attention_entropy = -(attention_weights * torch.log(attention_weights + 1e-10)).sum(dim=-1).mean()
                total_loss += self.entropy_loss_weight * attention_entropy
                loss_dict['attention_entropy'] = attention_entropy.item()

        loss_dict['total_loss'] = total_loss.item()
        return total_loss, loss_dict


class ContrastiveMILLoss(nn.Module):
    """
    Contrastive learning loss for MIL to learn discriminative features.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        base_loss_weight: float = 1.0,
        contrastive_loss_weight: float = 0.5
    ):
        """
        Initialize contrastive MIL loss.

        Args:
            temperature: Temperature for contrastive loss
            base_loss_weight: Weight for base classification loss
            contrastive_loss_weight: Weight for contrastive loss
        """
        super().__init__()
        self.temperature = temperature
        self.base_loss_weight = base_loss_weight
        self.contrastive_loss_weight = contrastive_loss_weight

    def forward(
        self,
        slide_logits: torch.Tensor,
        slide_embeddings: torch.Tensor,
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute contrastive MIL loss.

        Args:
            slide_logits: Slide predictions [B, num_classes]
            slide_embeddings: Slide embeddings [B, dim]
            targets: Slide labels [B]

        Returns:
            Total loss and loss components dict
        """
        # Classification loss
        cls_loss = F.cross_entropy(slide_logits, targets)
        total_loss = self.base_loss_weight * cls_loss

        loss_dict = {'cls_loss': cls_loss.item()}

        # Contrastive loss
        if self.contrastive_loss_weight > 0:
            # Normalize embeddings
            slide_embeddings = F.normalize(slide_embeddings, dim=-1)

            # Compute similarity matrix
            similarity = torch.matmul(slide_embeddings, slide_embeddings.t()) / self.temperature

            # Create positive/negative masks
            labels_matrix = targets.unsqueeze(0) == targets.unsqueeze(1)
            labels_matrix.fill_diagonal_(False)  # Remove self-similarity

            # Compute contrastive loss (InfoNCE)
            if labels_matrix.any():
                exp_sim = torch.exp(similarity)

                # Positive pairs
                pos_sim = (exp_sim * labels_matrix).sum(dim=1)

                # All pairs (excluding self)
                mask = ~torch.eye(len(targets), dtype=torch.bool, device=targets.device)
                all_sim = (exp_sim * mask).sum(dim=1)

                # Contrastive loss
                contrastive_loss = -torch.log(pos_sim / (all_sim + 1e-10) + 1e-10).mean()
                total_loss += self.contrastive_loss_weight * contrastive_loss
                loss_dict['contrastive_loss'] = contrastive_loss.item()

        loss_dict['total_loss'] = total_loss.item()
        return total_loss, loss_dict


class CoxPHLoss(nn.Module):
    """
    Cox Proportional Hazards loss for survival prediction.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        risk_scores: torch.Tensor,
        survival_times: torch.Tensor,
        event_indicators: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Cox PH loss.

        Args:
            risk_scores: Predicted risk scores [B]
            survival_times: Observed survival times [B]
            event_indicators: Event indicators (1 = event, 0 = censored) [B]

        Returns:
            Cox PH loss
        """
        # Sort by survival time
        sort_idx = torch.argsort(survival_times, descending=True)
        risk_scores = risk_scores[sort_idx]
        event_indicators = event_indicators[sort_idx]

        # Compute log risk
        log_risk = F.log_softmax(risk_scores, dim=0)

        # Cumulative sum for risk set
        risk_set_sum = torch.logcumsumexp(risk_scores.flip(0), dim=0).flip(0)

        # Cox partial likelihood
        uncensored_likelihood = risk_scores - risk_set_sum
        censored_likelihood = uncensored_likelihood * event_indicators
        neg_log_likelihood = -censored_likelihood.sum() / event_indicators.sum().clamp(min=1)

        return neg_log_likelihood


def create_loss_function(
    loss_type: str = 'cross_entropy',
    num_classes: int = 2,
    **kwargs
) -> nn.Module:
    """
    Factory function to create loss function.

    Args:
        loss_type: Type of loss ('cross_entropy', 'focal', 'instance_mil', etc.)
        num_classes: Number of classes
        **kwargs: Additional arguments for loss function

    Returns:
        Loss function instance
    """
    if loss_type == 'cross_entropy':
        return nn.CrossEntropyLoss(**kwargs)
    elif loss_type == 'focal':
        return FocalLoss(**kwargs)
    elif loss_type == 'instance_mil':
        return InstanceMILLoss(**kwargs)
    elif loss_type == 'max_pooling_mil':
        return MaxPoolingMILLoss(**kwargs)
    elif loss_type == 'attention_mil':
        return AttentionBasedMILLoss(**kwargs)
    elif loss_type == 'contrastive_mil':
        return ContrastiveMILLoss(**kwargs)
    elif loss_type == 'cox_ph':
        return CoxPHLoss(**kwargs)
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")
