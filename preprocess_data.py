#!/usr/bin/env python3
"""
Data preprocessing script for WSI datasets.
"""
#!/usr/bin/env python3
"""
Script for preprocessing WSI data.
"""

import os
# === ADD THIS LINE ===
os.add_dll_directory(r"C:\openslide\bin")
# =======================

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from hierarchical_mil.data.wsi_preprocessing import preprocess_dataset
from hierarchical_mil.data.dataset import prepare_dataset_from_wsi, create_label_file_template
from hierarchical_mil.utils.utils import setup_logging


def main():
    parser = argparse.ArgumentParser(description='Preprocess WSI dataset')
    parser.add_argument('--input_dir', type=str, required=True, 
                        help='Directory containing WSI files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save processed patches')
    parser.add_argument('--labels_file', type=str, required=True,
                        help='CSV file with slide IDs and labels')
    parser.add_argument('--patch_size', type=int, default=256,
                        help='Size of patches to extract')
    parser.add_argument('--stride', type=int, default=None,
                        help='Stride for patch extraction (default: patch_size)')
    parser.add_argument('--level', type=int, default=0,
                        help='Magnification level to extract patches from')
    parser.add_argument('--background_threshold', type=float, default=0.7,
                        help='Background detection threshold (0-1)')
    parser.add_argument('--file_extensions', nargs='+', 
                        default=['.svs', '.tif', '.tiff', '.ndpi'],
                        help='Supported file extensions')
    parser.add_argument('--create_labels_template', action='store_true',
                        help='Create a labels file template and exit')
    parser.add_argument('--slide_ids', nargs='+',
                        help='Slide IDs for template (required with --create_labels_template)')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(level='INFO')
    
    # Create labels template if requested
    if args.create_labels_template:
        if not args.slide_ids:
            logger.error("--slide_ids is required when using --create_labels_template")
            sys.exit(1)
        
        # Create dummy labels (you should replace these with actual labels)
        labels = [0] * len(args.slide_ids)  # All negative by default
        
        create_label_file_template(args.labels_file, args.slide_ids, labels)
        logger.info(f"Labels template created at {args.labels_file}")
        logger.info("Please edit the labels file to add correct labels for each slide.")
        return
    
    # Check input directory
    if not os.path.exists(args.input_dir):
        logger.error(f"Input directory not found: {args.input_dir}")
        sys.exit(1)
    
    # Check labels file
    if not os.path.exists(args.labels_file):
        logger.error(f"Labels file not found: {args.labels_file}")
        logger.info("Use --create_labels_template to create a template labels file")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info("Starting WSI preprocessing...")
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Labels file: {args.labels_file}")
    logger.info(f"Patch size: {args.patch_size}")
    logger.info(f"Stride: {args.stride or args.patch_size}")
    logger.info(f"Level: {args.level}")
    logger.info(f"Background threshold: {args.background_threshold}")
    
    # Preprocess dataset
    try:
        prepare_dataset_from_wsi(
            wsi_dir=args.input_dir,
            labels_file=args.labels_file,
            output_dir=args.output_dir,
            patch_size=args.patch_size,
            stride=args.stride,
            level=args.level,
            background_threshold=args.background_threshold,
            file_extensions=args.file_extensions
        )
        
        logger.info("Preprocessing completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during preprocessing: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()