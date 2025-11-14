#!/usr/bin/env python3
"""
Main training script for hierarchical MIL transformer.
"""
import os
os.add_dll_directory(r"C:\openslide\openslide-bin-4.0.0.8-windows-x64\bin")

import os
import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

import torch
import torch.nn as nn
from hierarchical_mil.utils.utils import (
    set_seed, setup_logging, load_config, save_config, 
    get_device, create_output_directory, save_model_summary,
    validate_config, create_default_config
)
from hierarchical_mil.data.dataset import WSIDataset, create_data_loaders
from hierarchical_mil.models.hierarchical_mil import create_hierarchical_mil_transformer
from hierarchical_mil.training.trainer import MILTrainer, create_optimizer, create_scheduler


def main():
    parser = argparse.ArgumentParser(description='Train Hierarchical MIL Transformer')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--train_data_dir', type=str, help='Training data directory')
    parser.add_argument('--train_labels_file', type=str, help='Training labels file')
    parser.add_argument('--val_data_dir', type=str, help='Validation data directory')
    parser.add_argument('--val_labels_file', type=str, help='Validation labels file')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--experiment_name', type=str, default='hierarchical_mil', help='Experiment name')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--create_config', action='store_true', help='Create default config file and exit')
    
    args = parser.parse_args()
    
    # Create default config if requested
    if args.create_config:
        default_config = create_default_config()
        config_path = 'config.yaml'
        save_config(default_config, config_path)
        print(f"Default configuration saved to {config_path}")
        return
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
        config = validate_config(config)
    else:
        config = create_default_config()
    
    # Override config with command line arguments
    if args.train_data_dir:
        config['data']['train_data_dir'] = args.train_data_dir
    if args.train_labels_file:
        config['data']['train_labels_file'] = args.train_labels_file
    if args.val_data_dir:
        config['data']['val_data_dir'] = args.val_data_dir
    if args.val_labels_file:
        config['data']['val_labels_file'] = args.val_labels_file
    if args.output_dir != 'outputs':
        config['experiment']['output_dir'] = args.output_dir
    if args.experiment_name != 'hierarchical_mil':
        config['experiment']['name'] = args.experiment_name
    if args.device != 'auto':
        config['experiment']['device'] = args.device
    if args.seed != 42:
        config['experiment']['seed'] = args.seed
    
    # Create output directory
    output_dir = create_output_directory(
        config['experiment']['output_dir'],
        config['experiment']['name']
    )
    
    # Setup logging
    log_file = os.path.join(output_dir, 'training.log')
    logger = setup_logging(log_file, config['experiment']['log_level'])
    
    # Set random seed
    set_seed(config['experiment']['seed'])
    logger.info(f"Random seed set to {config['experiment']['seed']}")
    
    # Get device
    if config['experiment']['device'] == 'auto':
        device = get_device()
    else:
        device = torch.device(config['experiment']['device'])
    logger.info(f"Using device: {device}")
    
    # Save configuration
    config_save_path = os.path.join(output_dir, 'config.yaml')
    save_config(config, config_save_path)
    logger.info(f"Configuration saved to {config_save_path}")
    
    # Check data directories
    train_data_dir = os.path.join(project_root, config['data']['train_data_dir'])
    train_labels_file = os.path.join(project_root, config['data']['train_labels_file'])
    
    if not os.path.exists(train_data_dir):
        logger.error(f"Training data directory not found: {train_data_dir}")
        sys.exit(1)
    
    if not os.path.exists(train_labels_file):
        logger.error(f"Training labels file not found: {train_labels_file}")
        sys.exit(1)
    
    # Create datasets
    logger.info("Creating datasets...")
    
    train_dataset = WSIDataset(
        data_dir=train_data_dir,
        labels_file=train_labels_file,
        patches_per_region=config['data']['patches_per_region'],
        regions_per_slide=config['data']['regions_per_slide'],
        patch_size=config['data']['patch_size'],
        cache_patches=config['data']['cache_patches'],
        preload_patches=config['data']['preload_patches'],
        augmentation_prob=config['data']['augmentation_prob']
    )
    
    logger.info(f"Training dataset: {len(train_dataset)} samples")
    
    val_dataset = None
    if 'val_data_dir' in config['data'] and os.path.exists(config['data']['val_data_dir']):
        val_dataset = WSIDataset(
            data_dir=os.path.join(project_root, config['data']['val_data_dir']),
            labels_file=os.path.join(project_root, config['data']['val_labels_file']),
            patches_per_region=config['data']['patches_per_region'],
            regions_per_slide=config['data']['regions_per_slide'],
            patch_size=config['data']['patch_size'],
            cache_patches=config['data']['cache_patches'],
            preload_patches=False,  # Don't preload validation data
            augmentation_prob=0.0  # No augmentation for validation
        )
        logger.info(f"Validation dataset: {len(val_dataset)} samples")
    
    # Create data loaders
    logger.info("Creating data loaders...")
    data_loaders = create_data_loaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=config['training']['batch_size'],
        num_workers=0,
        pin_memory=False,
        shuffle_train=True
    )
    
    # Create model
    logger.info("Creating model...")
    model = create_hierarchical_mil_transformer(
        config=config['model'],
        num_classes=train_dataset.label_to_idx.__len__()
    )
    
    # Log model information
    from hierarchical_mil.utils.utils import count_parameters, format_parameter_count
    param_counts = count_parameters(model)
    logger.info(f"Model created with {format_parameter_count(param_counts['total'])} parameters "
                f"({format_parameter_count(param_counts['trainable'])} trainable)")
    
    # Save model summary
    model_summary_path = os.path.join(output_dir, 'model_summary.json')
    save_model_summary(model, model_summary_path)
    
    # Create optimizer
    optimizer = create_optimizer(
        model=model,
        optimizer_type=config['training']['optimizer'],
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Create scheduler
    scheduler = create_scheduler(
        optimizer=optimizer,
        scheduler_type=config['training']['scheduler'],
        **config['training'].get('scheduler_params', {})
    )
    
    # Create loss function
    criterion = nn.CrossEntropyLoss()
    
    # Create trainer
    logger.info("Creating trainer...")
    trainer = MILTrainer(
        model=model,
        train_loader=data_loaders['train'],
        val_loader=data_loaders.get('val'),
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=str(device),
        output_dir=output_dir,
        log_interval=config['experiment']['log_interval'],
        save_interval=config['experiment']['save_interval'],
        max_epochs=config['training']['max_epochs'],
        early_stopping_patience=config['training']['early_stopping_patience'],
        use_tensorboard=config['experiment']['use_tensorboard'],
        gradient_clip_val=config['training']['gradient_clip_val'],
        accumulation_steps=config['training']['accumulation_steps']
    )
    
    # Start training
    logger.info("Starting training...")
    trainer.train()
    
    # Test model if test data is available
    if 'test_data_dir' in config['data'] and os.path.exists(config['data']['test_data_dir']):
        logger.info("Running final evaluation on test set...")
        
        test_dataset = WSIDataset(
            data_dir=config['data']['test_data_dir'],
            labels_file=config['data']['test_labels_file'],
            patches_per_region=config['data']['patches_per_region'],
            regions_per_slide=config['data']['regions_per_slide'],
            patch_size=config['data']['patch_size'],
            cache_patches=False,
            preload_patches=False,
            augmentation_prob=0.0
        )
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=config['training']['num_workers'],
            pin_memory=True,
            drop_last=False
        )
        
        trainer.test_loader = test_loader
        test_metrics = trainer.test_model(trainer.best_model_path)
        
        logger.info("Test Results:")
        for metric, value in test_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
    
    logger.info("Training completed successfully!")
    logger.info(f"Output directory: {output_dir}")


if __name__ == '__main__':
    main()