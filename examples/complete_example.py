"""
Example script showing how to use the hierarchical MIL transformer.
This demonstrates the complete workflow from data preparation to training.
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
import tempfile
import shutil

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from hierarchical_mil.utils.utils import set_seed, create_default_config, save_config
from hierarchical_mil.data.wsi_preprocessing import WSITiler
from hierarchical_mil.data.dataset import WSIDataset, create_data_loaders, create_label_file_template
from hierarchical_mil.models.hierarchical_mil import create_hierarchical_mil_transformer
from hierarchical_mil.training.trainer import MILTrainer, create_optimizer, create_scheduler
import torch.nn as nn


def create_dummy_dataset(output_dir: str, num_slides: int = 10):
    """
    Create a dummy dataset for demonstration purposes.
    In practice, you would use real WSI data.
    """
    print("Creating dummy dataset...")
    
    # Create directories
    os.makedirs(output_dir, exist_ok=True)
    
    # Create dummy slide images and extract patches
    slide_ids = []
    labels = []
    
    tiler = WSITiler(
        patch_size=256,
        stride=256,
        background_threshold=0.9  # Low threshold to include most patches
    )
    
    for i in range(num_slides):
        slide_id = f"slide_{i:03d}"
        slide_ids.append(slide_id)
        
        # Random label (0 or 1)
        label = np.random.randint(0, 2)
        labels.append(label)
        
        # Create dummy WSI (1024x1024 RGB image)
        dummy_wsi = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
        
        # Add some structure to make patches more realistic
        # Create tissue-like regions
        center_x, center_y = 512, 512
        for _ in range(5):  # Add 5 circular regions
            cx = np.random.randint(200, 824)
            cy = np.random.randint(200, 824)
            radius = np.random.randint(50, 150)
            
            y, x = np.ogrid[:1024, :1024]
            mask = (x - cx)**2 + (y - cy)**2 <= radius**2
            
            # Add tissue-like color
            tissue_color = np.random.randint(100, 200, 3)
            dummy_wsi[mask] = tissue_color
        
        # Save as temporary image and extract patches
        temp_image_path = f"/tmp/temp_slide_{i}.png"
        Image.fromarray(dummy_wsi).save(temp_image_path)
        
        # Extract patches
        output_path = os.path.join(output_dir, f"{slide_id}_patches.h5")
        tiler.save_patches_to_h5(temp_image_path, output_path)
        
        # Clean up temporary file
        os.remove(temp_image_path)
    
    # Create labels file
    labels_file = os.path.join(output_dir, "labels.csv")
    create_label_file_template(labels_file, slide_ids, labels)
    
    print(f"Created dummy dataset with {num_slides} slides")
    print(f"Data saved to: {output_dir}")
    print(f"Labels saved to: {labels_file}")
    
    return slide_ids, labels


def main():
    """Main example workflow."""
    print("Hierarchical MIL Transformer Example")
    print("====================================")
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Create temporary directory for demo
    temp_dir = tempfile.mkdtemp()
    print(f"Using temporary directory: {temp_dir}")
    
    try:
        # Step 1: Create dummy dataset
        data_dir = os.path.join(temp_dir, "data")
        slide_ids, labels = create_dummy_dataset(data_dir, num_slides=20)
        
        # Split data into train/val
        train_size = int(0.8 * len(slide_ids))
        train_slide_ids = slide_ids[:train_size]
        train_labels = labels[:train_size]
        val_slide_ids = slide_ids[train_size:]
        val_labels = labels[train_size:]
        
        # Create train/val label files
        train_labels_file = os.path.join(data_dir, "train_labels.csv")
        val_labels_file = os.path.join(data_dir, "val_labels.csv")
        
        create_label_file_template(train_labels_file, train_slide_ids, train_labels)
        create_label_file_template(val_labels_file, val_slide_ids, val_labels)
        
        print(f"Train samples: {len(train_slide_ids)}")
        print(f"Validation samples: {len(val_slide_ids)}")
        
        # Step 2: Create configuration
        config = create_default_config()
        
        # Adjust config for demo (smaller model, fewer epochs)
        config['model'].update({
            'backbone_type': 'resnet18',  # Smaller model
            'patch_feature_dim': 128,
            'region_dim': 128,
            'slide_dim': 128,
            'patches_per_region': 16,  # Fewer patches for speed
            'regions_per_slide': 8,
            'region_attention_layers': 1,
            'slide_attention_layers': 2
        })
        
        config['data'].update({
            'train_data_dir': data_dir,
            'val_data_dir': data_dir,
            'train_labels_file': train_labels_file,
            'val_labels_file': val_labels_file,
            'patches_per_region': 16,
            'regions_per_slide': 8,
            'cache_patches': True,
            'preload_patches': False
        })
        
        config['training'].update({
            'batch_size': 2,  # Small batch size
            'max_epochs': 5,  # Few epochs for demo
            'num_workers': 0,  # Avoid multiprocessing issues
            'early_stopping_patience': 3
        })
        
        config['experiment'].update({
            'name': 'demo_experiment',
            'output_dir': os.path.join(temp_dir, 'outputs'),
            'use_tensorboard': False  # Disable for demo
        })
        
        # Save config
        config_path = os.path.join(temp_dir, "config.yaml")
        save_config(config, config_path)
        print(f"Configuration saved to: {config_path}")
        
        # Step 3: Create datasets
        print("\nCreating datasets...")
        
        train_dataset = WSIDataset(
            data_dir=data_dir,
            labels_file=train_labels_file,
            patches_per_region=config['data']['patches_per_region'],
            regions_per_slide=config['data']['regions_per_slide'],
            patch_size=config['data']['patch_size'],
            cache_patches=config['data']['cache_patches'],
            preload_patches=config['data']['preload_patches']
        )
        
        val_dataset = WSIDataset(
            data_dir=data_dir,
            labels_file=val_labels_file,
            patches_per_region=config['data']['patches_per_region'],
            regions_per_slide=config['data']['regions_per_slide'],
            patch_size=config['data']['patch_size'],
            cache_patches=False,
            preload_patches=False
        )
        
        print(f"Training dataset: {len(train_dataset)} samples")
        print(f"Validation dataset: {len(val_dataset)} samples")
        
        # Step 4: Create data loaders
        data_loaders = create_data_loaders(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            batch_size=config['training']['batch_size'],
            num_workers=config['training']['num_workers']
        )
        
        # Step 5: Create model
        print("\nCreating model...")
        
        model = create_hierarchical_mil_transformer(
            config=config['model'],
            num_classes=len(train_dataset.label_to_idx)
        )
        
        # Print model info
        from hierarchical_mil.utils.utils import count_parameters, format_parameter_count
        param_counts = count_parameters(model)
        print(f"Model parameters: {format_parameter_count(param_counts['total'])} "
              f"({format_parameter_count(param_counts['trainable'])} trainable)")
        
        # Step 6: Test model forward pass
        print("\nTesting model forward pass...")
        
        # Get a sample batch
        sample_batch = next(iter(data_loaders['train']))
        
        with torch.no_grad():
            outputs = model(
                patches=sample_batch['patches'],
                region_masks=sample_batch['region_masks'],
                slide_masks=sample_batch['slide_masks'],
                return_attention=True,
                return_embeddings=True
            )
        
        print("Forward pass successful!")
        print(f"Output shapes:")
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
        
        # Step 7: Create training components
        print("\nSetting up training...")
        
        optimizer = create_optimizer(
            model=model,
            optimizer_type=config['training']['optimizer'],
            learning_rate=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        scheduler = create_scheduler(
            optimizer=optimizer,
            scheduler_type=config['training']['scheduler'],
            **config['training'].get('scheduler_params', {})
        )
        
        criterion = nn.CrossEntropyLoss()
        
        # Step 8: Create trainer and start training
        print("\nStarting training...")
        
        trainer = MILTrainer(
            model=model,
            train_loader=data_loaders['train'],
            val_loader=data_loaders['val'],
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device='cpu',  # Use CPU for demo
            output_dir=config['experiment']['output_dir'],
            max_epochs=config['training']['max_epochs'],
            early_stopping_patience=config['training']['early_stopping_patience'],
            use_tensorboard=config['experiment']['use_tensorboard']
        )
        
        # Run training
        trainer.train()
        
        print("\nTraining completed!")
        print(f"Best validation score: {trainer.best_val_score:.4f}")
        
        # Step 9: Test inference
        print("\nTesting inference...")
        
        model.eval()
        test_sample = sample_batch
        
        with torch.no_grad():
            outputs = model(
                patches=test_sample['patches'],
                region_masks=test_sample['region_masks'],
                slide_masks=test_sample['slide_masks']
            )
            
            predictions = outputs['logits'].argmax(dim=1)
            probabilities = outputs['probabilities']
        
        print("Inference results:")
        for i in range(len(predictions)):
            true_label = test_sample['label'][i].item()
            pred_label = predictions[i].item()
            confidence = probabilities[i].max().item()
            slide_id = test_sample['slide_id'][i]
            
            print(f"  {slide_id}: True={true_label}, Pred={pred_label}, Confidence={confidence:.3f}")
        
        print("\nExample completed successfully!")
        print(f"Output directory: {config['experiment']['output_dir']}")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up temporary directory
        try:
            shutil.rmtree(temp_dir)
            print(f"\nCleaned up temporary directory: {temp_dir}")
        except:
            print(f"\nWarning: Could not clean up temporary directory: {temp_dir}")


if __name__ == '__main__':
    main()