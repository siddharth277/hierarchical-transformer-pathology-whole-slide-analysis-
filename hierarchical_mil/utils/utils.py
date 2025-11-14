"""
Utility functions for hierarchical MIL transformer.
"""

import os
import yaml
import json
import torch
import numpy as np
import random
from typing import Dict, Any, Optional
import logging


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(log_file: Optional[str] = None, level: str = 'INFO') -> logging.Logger:
    """Setup logging configuration."""
    logger = logging.getLogger('hierarchical_mil')
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file is not None:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML or JSON file."""
    with open(config_path, 'r') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            config = yaml.safe_load(f)
        elif config_path.endswith('.json'):
            config = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path}")
    
    return config


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """Save configuration to file."""
    output_dir = os.path.dirname(output_path)
    if output_dir:  # Only create directory if there is one
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_path, 'w') as f:
        if output_path.endswith('.yaml') or output_path.endswith('.yml'):
            yaml.dump(config, f, default_flow_style=False, indent=2)
        elif output_path.endswith('.json'):
            json.dump(config, f, indent=2)
        else:
            raise ValueError(f"Unsupported config file format: {output_path}")


def get_device(device: Optional[str] = None) -> torch.device:
    """Get the appropriate device for computation."""
    if device is not None:
        return torch.device(device)
    
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Count the number of parameters in a model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }


def format_parameter_count(param_count: int) -> str:
    """Format parameter count in a human-readable way."""
    if param_count >= 1e9:
        return f"{param_count / 1e9:.2f}B"
    elif param_count >= 1e6:
        return f"{param_count / 1e6:.2f}M"
    elif param_count >= 1e3:
        return f"{param_count / 1e3:.2f}K"
    else:
        return str(param_count)


def create_output_directory(base_dir: str, experiment_name: str) -> str:
    """Create output directory with timestamp."""
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"{experiment_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def save_model_summary(model: torch.nn.Module, output_path: str) -> None:
    """Save model summary to file."""
    param_counts = count_parameters(model)
    
    summary = {
        'model_class': model.__class__.__name__,
        'total_parameters': param_counts['total'],
        'trainable_parameters': param_counts['trainable'],
        'non_trainable_parameters': param_counts['non_trainable'],
        'total_parameters_formatted': format_parameter_count(param_counts['total']),
        'trainable_parameters_formatted': format_parameter_count(param_counts['trainable'])
    }
    
    # Add model-specific information
    if hasattr(model, 'patch_feature_dim'):
        summary['patch_feature_dim'] = model.patch_feature_dim
    if hasattr(model, 'region_dim'):
        summary['region_dim'] = model.region_dim
    if hasattr(model, 'slide_dim'):
        summary['slide_dim'] = model.slide_dim
    if hasattr(model, 'num_classes'):
        summary['num_classes'] = model.num_classes
    
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and fill in default values for configuration."""
    default_config = {
        'model': {
            'backbone_type': 'resnet50',
            'patch_feature_dim': 512,
            'pretrained_backbone': True,
            'freeze_backbone': False,
            'region_dim': 512,
            'patches_per_region': 64,
            'region_attention_heads': 8,
            'region_attention_layers': 2,
            'slide_dim': 512,
            'regions_per_slide': 32,
            'slide_attention_heads': 8,
            'slide_attention_layers': 4,
            'num_classes': 2,
            'dropout': 0.1,
            'use_gradient_checkpointing': False
        },
        'data': {
            'patch_size': 256,
            'stride': None,
            'level': 0,
            'background_threshold': 0.7,
            'patches_per_region': 64,
            'regions_per_slide': 32,
            'augmentation_prob': 0.0
        },
        'training': {
            'batch_size': 4,
            'num_workers': 4,
            'max_epochs': 100,
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'optimizer': 'adam',
            'scheduler': 'plateau',
            'early_stopping_patience': 10,
            'gradient_clip_val': None,
            'accumulation_steps': 1
        },
        'experiment': {
            'seed': 42,
            'device': 'auto',
            'output_dir': 'outputs',
            'log_interval': 10,
            'save_interval': 1,
            'use_tensorboard': True
        }
    }
    
    # Merge configurations
    validated_config = {}
    for section in default_config:
        validated_config[section] = {**default_config[section], **config.get(section, {})}
    
    return validated_config


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """Early stopping utility."""
    
    def __init__(self, patience=10, min_delta=0, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self._is_better(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
    
    def _is_better(self, score):
        if self.mode == 'max':
            return score > self.best_score + self.min_delta
        else:
            return score < self.best_score - self.min_delta


def visualize_attention(
    attention_weights: torch.Tensor,
    save_path: str,
    title: str = "Attention Weights"
) -> None:
    """Visualize attention weights."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Convert to numpy
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.detach().cpu().numpy()
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(attention_weights, cmap='viridis', cbar=True)
    plt.title(title)
    plt.xlabel('Key')
    plt.ylabel('Query')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def memory_usage():
    """Get current memory usage."""
    if torch.cuda.is_available():
        return {
            'allocated': torch.cuda.memory_allocated() / 1024**2,  # MB
            'cached': torch.cuda.memory_reserved() / 1024**2,      # MB
            'max_allocated': torch.cuda.max_memory_allocated() / 1024**2  # MB
        }
    else:
        return {'message': 'CUDA not available'}


def cleanup_memory():
    """Clean up GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def create_default_config() -> Dict[str, Any]:
    """Create a default configuration dictionary."""
    return {
        'model': {
            'backbone_type': 'resnet50',
            'patch_feature_dim': 512,
            'pretrained_backbone': True,
            'freeze_backbone': False,
            'region_dim': 512,
            'patches_per_region': 64,
            'region_attention_heads': 8,
            'region_attention_layers': 2,
            'slide_dim': 512,
            'regions_per_slide': 32,
            'slide_attention_heads': 8,
            'slide_attention_layers': 4,
            'num_classes': 2,
            'dropout': 0.1,
            'use_gradient_checkpointing': False
        },
        'data': {
            'train_data_dir': 'data/train',
            'val_data_dir': 'data/val',
            'test_data_dir': 'data/test',
            'train_labels_file': 'data/train_labels.csv',
            'val_labels_file': 'data/val_labels.csv',
            'test_labels_file': 'data/test_labels.csv',
            'patch_size': 256,
            'stride': None,
            'level': 0,
            'background_threshold': 0.7,
            'patches_per_region': 64,
            'regions_per_slide': 32,
            'augmentation_prob': 0.0,
            'cache_patches': True,
            'preload_patches': False
        },
        'training': {
            'batch_size': 4,
            'num_workers': 4,
            'max_epochs': 100,
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'optimizer': 'adam',
            'scheduler': 'plateau',
            'scheduler_params': {
                'patience': 5,
                'factor': 0.5
            },
            'early_stopping_patience': 10,
            'gradient_clip_val': None,
            'accumulation_steps': 1
        },
        'experiment': {
            'name': 'hierarchical_mil_experiment',
            'seed': 42,
            'device': 'auto',
            'output_dir': 'outputs',
            'log_interval': 10,
            'save_interval': 1,
            'use_tensorboard': True,
            'log_level': 'INFO'
        }
    }