#!/usr/bin/env python3
"""
Run inference and evaluation on a trained hierarchical MIL model.
"""

import os
os.add_dll_directory(r"C:\openslide\openslide-bin-4.0.0.8-windows-x64\bin")
# === FIX FOR OPENSLIDE (if you ever use .svs files) ===
try:
    os.add_dll_directory(r"C:\openslide\openslide-bin-4.0.0.8-windows-x64\bin")
except AttributeError:
    pass
# ======================================================

import argparse
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from hierarchical_mil.utils.utils import (
    set_seed, setup_logging, load_config, get_device,
    validate_config
)
from hierarchical_mil.data.dataset import WSIDataset
from hierarchical_mil.models.hierarchical_mil import create_hierarchical_mil_transformer

def main():
    parser = argparse.ArgumentParser(description='Inference for Hierarchical MIL Transformer')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model checkpoint (.pth file)')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file (config.yaml)')
    parser.add_argument('--data_dir', type=str, required=True, help='Test data directory')
    parser.add_argument('--labels_file', type=str, required=True, help='Test labels file')
    parser.add_argument('--output_file', type=str, default='predictions.csv', help='File to save predictions')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference (default: 1)')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (auto, cuda, cpu)')
    parser.add_argument('--return_attention', action='store_true', help='Save attention weights')
    
    args = parser.parse_args()

    logger = setup_logging()

    if args.device == 'auto':
        device = get_device()
    else:
        device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    config = load_config(args.config)
    config = validate_config(config)
    
    # === FIX FOR PATHS ===
    # Use absolute paths to prevent FileNotFoundError
    data_dir_abs = os.path.join(project_root, args.data_dir)
    labels_file_abs = os.path.join(project_root, args.labels_file)

    logger.info("Creating dataset...")
    test_dataset = WSIDataset(
        data_dir=data_dir_abs,
        labels_file=labels_file_abs,
        patches_per_region=config['data']['patches_per_region'],
        regions_per_slide=config['data']['regions_per_slide'],
        patch_size=config['data']['patch_size'],
        cache_patches=False,
        preload_patches=False,
        augmentation_prob=0.0
    )

    data_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,  # Use 0 for Windows compatibility
        pin_memory=False # Use False to save VRAM
    )

    logger.info("Creating model...")
    
    # === FIX FOR MODEL MISMATCH ===
    # We hard-code num_classes=2 to match the trained model.
    model = create_hierarchical_mil_transformer(
        config=config['model'],
        num_classes=2  # <-- THIS IS THE FIX
    )
    # ==============================

    logger.info(f"Loading model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    all_labels = []
    all_predictions = []
    all_slide_ids = []
    
    logger.info("Running inference...")
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Inference'):
            patches = batch['patches'].to(device, non_blocking=True)
            region_masks = batch['region_masks'].to(device, non_blocking=True)
            slide_masks = batch['slide_masks'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)
            
            # === FIX FOR ARGUMENT TYPOS ===
            outputs = model(
                patches=patches,
                region_masks=region_masks, # <-- Added 's'
                slide_masks=slide_masks, # <-- Added 's'
                return_embeddings=False,
                return_attention=args.return_attention
            )
            # ==============================
            
            # === FIX FOR DICTIONARY OUTPUT ===
            if isinstance(outputs, dict):
                logits = outputs['logits']
            else:
                logits = outputs
            
            if config['model']['task_type'] == 'survival':
                predictions = logits
            else:
                predictions = torch.argmax(logits, dim=1) # Use extracted logits
            # ===============================

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_slide_ids.extend(batch['slide_id'])

    predictions_df = pd.DataFrame({
        'slide_id': all_slide_ids,
        'label': all_labels,
        'prediction': all_predictions
    })
    predictions_df.to_csv(args.output_file, index=False)
    logger.info(f"Predictions saved to {args.output_file}")

    # === FIX FOR REPORTING ===
    # Manually define all known labels (0 and 1)
    all_known_labels = [0, 1]
    target_names = ['Normal', 'Tumor'] 
    # =========================
    
    accuracy = accuracy_score(all_labels, all_predictions)
    logger.info(f"Accuracy: {accuracy:.4f}")
    
    logger.info("Classification Report:")
    report = classification_report(
        all_labels,
        all_predictions,
        labels=all_known_labels,    # <-- Use the fix
        target_names=target_names,  # <-- Use the fix
        zero_division=0
    )
    logger.info(f"\n{report}")
    
    logger.info("Confusion Matrix:")
    cm = confusion_matrix(
        all_labels, 
        all_predictions, 
        labels=all_known_labels # <-- Use the fix
    )
    logger.info(f"\n{cm}")
    
    try:
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='g', 
            cmap='Blues',
            xticklabels=target_names,
            yticklabels=target_names
        )
        plt.xlabel('Predicted')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.savefig('confusion_matrix.png')
        logger.info("Confusion matrix plot saved to confusion_matrix.png")
    except Exception as e:
        logger.warning(f"Could not plot confusion matrix: {e}")

if __name__ == '__main__':
    main()