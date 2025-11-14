"""
Visualization utilities for hierarchical MIL.
Includes attention heatmaps, patch visualization, and interpretability tools.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, List, Dict
import cv2
from PIL import Image
import os


class AttentionVisualizer:
    """
    Visualizer for attention weights in hierarchical MIL.
    """

    def __init__(self, figsize: Tuple[int, int] = (15, 10), dpi: int = 300):
        """
        Initialize attention visualizer.

        Args:
            figsize: Figure size for plots
            dpi: DPI for saved figures
        """
        self.figsize = figsize
        self.dpi = dpi

    def visualize_slide_attention(
        self,
        attention_weights: np.ndarray,
        save_path: str,
        title: str = "Slide-Level Attention"
    ) -> None:
        """
        Visualize slide-level attention weights.

        Args:
            attention_weights: Attention weights [num_regions]
            save_path: Path to save visualization
            title: Plot title
        """
        plt.figure(figsize=self.figsize)

        # Create bar plot
        regions = np.arange(len(attention_weights))
        plt.bar(regions, attention_weights, color='steelblue', alpha=0.7)
        plt.xlabel('Region Index', fontsize=12)
        plt.ylabel('Attention Weight', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

    def visualize_region_attention(
        self,
        attention_weights: np.ndarray,
        save_path: str,
        title: str = "Region-Level Attention",
        num_regions: Optional[int] = None
    ) -> None:
        """
        Visualize region-level attention weights (patches within regions).

        Args:
            attention_weights: Attention weights [num_regions, num_patches]
            save_path: Path to save visualization
            title: Plot title
            num_regions: Number of regions to visualize (None = all)
        """
        if num_regions is not None:
            attention_weights = attention_weights[:num_regions]

        fig, axes = plt.subplots(
            nrows=min(4, len(attention_weights)),
            ncols=1,
            figsize=(self.figsize[0], min(12, len(attention_weights) * 3))
        )

        if len(attention_weights) == 1:
            axes = [axes]

        for idx, (ax, region_attn) in enumerate(zip(axes, attention_weights)):
            patches = np.arange(len(region_attn))
            ax.bar(patches, region_attn, color='coral', alpha=0.7)
            ax.set_xlabel('Patch Index', fontsize=10)
            ax.set_ylabel('Attention Weight', fontsize=10)
            ax.set_title(f'Region {idx} - Patch Attention', fontsize=11)
            ax.grid(axis='y', alpha=0.3)

        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.0)
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

    def visualize_hierarchical_attention(
        self,
        slide_attention: np.ndarray,
        region_attention: np.ndarray,
        save_path: str,
        title: str = "Hierarchical Attention"
    ) -> None:
        """
        Visualize both slide and region-level attention in one plot.

        Args:
            slide_attention: Slide-level attention [num_regions]
            region_attention: Region-level attention [num_regions, num_patches]
            save_path: Path to save visualization
            title: Plot title
        """
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 2, 2], width_ratios=[1, 1], hspace=0.3, wspace=0.3)

        # Slide-level attention (top, spanning both columns)
        ax_slide = fig.add_subplot(gs[0, :])
        regions = np.arange(len(slide_attention))
        ax_slide.bar(regions, slide_attention, color='steelblue', alpha=0.7)
        ax_slide.set_xlabel('Region Index', fontsize=12)
        ax_slide.set_ylabel('Attention Weight', fontsize=12)
        ax_slide.set_title('Slide-Level Attention', fontsize=13, fontweight='bold')
        ax_slide.grid(axis='y', alpha=0.3)

        # Region-level attention heatmap (bottom left)
        ax_heatmap = fig.add_subplot(gs[1:, 0])
        sns.heatmap(
            region_attention,
            cmap='YlOrRd',
            cbar=True,
            ax=ax_heatmap,
            cbar_kws={'label': 'Attention Weight'}
        )
        ax_heatmap.set_xlabel('Patch Index', fontsize=12)
        ax_heatmap.set_ylabel('Region Index', fontsize=12)
        ax_heatmap.set_title('Region-Level Attention Heatmap', fontsize=13, fontweight='bold')

        # Top regions detail (bottom right)
        ax_top = fig.add_subplot(gs[1:, 1])
        top_k = min(5, len(slide_attention))
        top_regions_idx = np.argsort(slide_attention)[-top_k:][::-1]

        for i, region_idx in enumerate(top_regions_idx):
            ax_top.plot(
                region_attention[region_idx],
                label=f'Region {region_idx} ({slide_attention[region_idx]:.3f})',
                marker='o',
                alpha=0.7
            )

        ax_top.set_xlabel('Patch Index', fontsize=12)
        ax_top.set_ylabel('Attention Weight', fontsize=12)
        ax_top.set_title(f'Top {top_k} Regions - Patch Attention', fontsize=13, fontweight='bold')
        ax_top.legend(loc='best', fontsize=9)
        ax_top.grid(alpha=0.3)

        plt.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()


class WSIHeatmapGenerator:
    """
    Generate spatial heatmaps on whole slide images.
    """

    def __init__(
        self,
        patch_size: int = 256,
        downsample_factor: int = 16,
        colormap: str = 'jet'
    ):
        """
        Initialize WSI heatmap generator.

        Args:
            patch_size: Size of patches
            downsample_factor: Downsample factor for visualization
            colormap: Matplotlib colormap name
        """
        self.patch_size = patch_size
        self.downsample_factor = downsample_factor
        self.colormap = plt.get_cmap(colormap)

    def generate_attention_heatmap(
        self,
        attention_scores: np.ndarray,
        coordinates: np.ndarray,
        slide_shape: Tuple[int, int],
        save_path: str,
        alpha: float = 0.5
    ) -> None:
        """
        Generate attention heatmap overlaid on WSI coordinates.

        Args:
            attention_scores: Attention scores for each patch [N]
            coordinates: Patch coordinates [N, 2] (x, y)
            slide_shape: Original slide shape (width, height)
            save_path: Path to save heatmap
            alpha: Transparency for overlay
        """
        # Create heatmap canvas
        heatmap_width = slide_shape[0] // self.downsample_factor
        heatmap_height = slide_shape[1] // self.downsample_factor

        heatmap = np.zeros((heatmap_height, heatmap_width), dtype=np.float32)
        count_map = np.zeros((heatmap_height, heatmap_width), dtype=np.float32)

        # Normalize attention scores
        attention_scores = (attention_scores - attention_scores.min()) / (attention_scores.max() - attention_scores.min() + 1e-10)

        # Fill heatmap
        patch_size_scaled = self.patch_size // self.downsample_factor

        for (x, y), score in zip(coordinates, attention_scores):
            x_scaled = int(x // self.downsample_factor)
            y_scaled = int(y // self.downsample_factor)

            x_end = min(x_scaled + patch_size_scaled, heatmap_width)
            y_end = min(y_scaled + patch_size_scaled, heatmap_height)

            heatmap[y_scaled:y_end, x_scaled:x_end] += score
            count_map[y_scaled:y_end, x_scaled:x_end] += 1

        # Average overlapping regions
        heatmap = np.divide(heatmap, count_map, where=count_map > 0)

        # Apply colormap
        heatmap_colored = self.colormap(heatmap)
        heatmap_colored = (heatmap_colored[:, :, :3] * 255).astype(np.uint8)

        # Save heatmap
        Image.fromarray(heatmap_colored).save(save_path)

    def overlay_heatmap_on_image(
        self,
        heatmap: np.ndarray,
        background_image: np.ndarray,
        save_path: str,
        alpha: float = 0.5
    ) -> None:
        """
        Overlay heatmap on background image.

        Args:
            heatmap: Heatmap array [H, W] or [H, W, 3]
            background_image: Background image [H, W, 3]
            save_path: Path to save result
            alpha: Heatmap transparency
        """
        # Ensure same size
        if heatmap.shape[:2] != background_image.shape[:2]:
            heatmap = cv2.resize(heatmap, (background_image.shape[1], background_image.shape[0]))

        # Convert heatmap to RGB if needed
        if len(heatmap.shape) == 2:
            heatmap = self.colormap(heatmap)[:, :, :3]
            heatmap = (heatmap * 255).astype(np.uint8)

        # Blend images
        overlay = cv2.addWeighted(background_image, 1 - alpha, heatmap, alpha, 0)

        # Save
        Image.fromarray(overlay).save(save_path)


class PatchVisualizer:
    """
    Visualize patches and their predictions.
    """

    def __init__(self, figsize: Tuple[int, int] = (20, 12), dpi: int = 150):
        """
        Initialize patch visualizer.

        Args:
            figsize: Figure size
            dpi: DPI for saved figures
        """
        self.figsize = figsize
        self.dpi = dpi

    def visualize_top_attended_patches(
        self,
        patches: np.ndarray,
        attention_scores: np.ndarray,
        save_path: str,
        top_k: int = 16,
        title: str = "Top Attended Patches"
    ) -> None:
        """
        Visualize patches with highest attention.

        Args:
            patches: Patches array [N, H, W, 3]
            attention_scores: Attention scores [N]
            save_path: Path to save visualization
            top_k: Number of top patches to show
            title: Plot title
        """
        # Get top-k patches
        top_idx = np.argsort(attention_scores)[-top_k:][::-1]

        # Calculate grid size
        cols = 4
        rows = (top_k + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=self.figsize)
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

        for i, ax in enumerate(axes):
            if i < len(top_idx):
                idx = top_idx[i]
                patch = patches[idx]

                # Handle different input formats
                if patch.dtype == np.float32 or patch.dtype == np.float64:
                    patch = (patch * 255).astype(np.uint8) if patch.max() <= 1.0 else patch.astype(np.uint8)

                ax.imshow(patch)
                ax.set_title(f'Rank {i+1}\nAttn: {attention_scores[idx]:.4f}', fontsize=10)
                ax.axis('off')
            else:
                ax.axis('off')

        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

    def visualize_patches_by_region(
        self,
        patches: np.ndarray,
        region_labels: np.ndarray,
        save_dir: str,
        patches_per_region: int = 8
    ) -> None:
        """
        Visualize patches organized by regions.

        Args:
            patches: All patches [N, H, W, 3]
            region_labels: Region assignment for each patch [N]
            save_dir: Directory to save visualizations
            patches_per_region: Number of patches to show per region
        """
        os.makedirs(save_dir, exist_ok=True)

        unique_regions = np.unique(region_labels)

        for region_id in unique_regions:
            region_mask = region_labels == region_id
            region_patches = patches[region_mask]

            # Sample patches if too many
            if len(region_patches) > patches_per_region:
                sample_idx = np.random.choice(len(region_patches), patches_per_region, replace=False)
                region_patches = region_patches[sample_idx]

            # Create visualization
            cols = 4
            rows = (len(region_patches) + cols - 1) // cols

            fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
            axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

            for i, ax in enumerate(axes):
                if i < len(region_patches):
                    patch = region_patches[i]
                    if patch.dtype == np.float32 or patch.dtype == np.float64:
                        patch = (patch * 255).astype(np.uint8) if patch.max() <= 1.0 else patch.astype(np.uint8)
                    ax.imshow(patch)
                    ax.axis('off')
                else:
                    ax.axis('off')

            plt.suptitle(f'Region {region_id}', fontsize=14, fontweight='bold')
            plt.tight_layout()

            save_path = os.path.join(save_dir, f'region_{region_id}.png')
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()


def create_confusion_matrix_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    normalize: bool = False,
    figsize: Tuple[int, int] = (10, 8),
    dpi: int = 300
) -> None:
    """
    Create and save confusion matrix plot.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save plot
        normalize: Whether to normalize confusion matrix
        figsize: Figure size
        dpi: DPI for saved figure
    """
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt='.2f' if normalize else 'd',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Proportion' if normalize else 'Count'}
    )

    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''), fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def create_roc_curve_plot(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    dpi: int = 300
) -> Dict[str, float]:
    """
    Create ROC curve plot for binary classification.

    Args:
        y_true: True labels
        y_scores: Predicted scores/probabilities
        save_path: Path to save plot
        figsize: Figure size
        dpi: DPI for saved figure

    Returns:
        Dictionary with AUC and other metrics
    """
    from sklearn.metrics import roc_curve, auc

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    return {'auc': roc_auc, 'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds}
