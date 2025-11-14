"""
WSI (Whole Slide Image) preprocessing and tiling utilities.
Handles opening, tiling, and preprocessing of gigapixel pathology slides.
"""

import os
import numpy as np
from typing import Tuple, List, Optional, Union
import cv2
from PIL import Image
import h5py
from tqdm import tqdm

try:
    import openslide
except ImportError:
    openslide = None
    print("Warning: openslide not available. WSI functionality will be limited.")


class WSITiler:
    """
    WSI tiling class for extracting patches from whole slide images.
    Supports multiple backends and efficient memory management.
    """
    
    def __init__(
        self,
        patch_size: int = 256,
        stride: Optional[int] = None,
        level: int = 0,
        background_threshold: float = 0.7,
        save_patches: bool = False,
        save_dir: Optional[str] = None
    ):
        """
        Initialize WSI tiler.
        
        Args:
            patch_size: Size of extracted patches (square)
            stride: Stride for patch extraction. If None, uses patch_size (non-overlapping)
            level: Magnification level to extract patches from
            background_threshold: Threshold for background detection (0-1)
            save_patches: Whether to save extracted patches to disk
            save_dir: Directory to save patches (required if save_patches=True)
        """
        self.patch_size = patch_size
        self.stride = stride or patch_size
        self.level = level
        self.background_threshold = background_threshold
        self.save_patches = save_patches
        self.save_dir = save_dir
        
        if self.save_patches and not self.save_dir:
            raise ValueError("save_dir must be provided when save_patches=True")
            
        if self.save_patches:
            os.makedirs(self.save_dir, exist_ok=True)
    
    def _is_background(self, patch: np.ndarray) -> bool:
        """
        Determine if a patch is mostly background.
        
        Args:
            patch: RGB patch as numpy array
            
        Returns:
            True if patch is background, False otherwise
        """
        # Convert to grayscale
        gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        
        # Calculate the ratio of white/light pixels
        white_ratio = np.sum(gray > 200) / gray.size
        
        return white_ratio > self.background_threshold
    
    def _open_slide(self, slide_path: str):
        """Open slide using available backend."""
        if openslide and slide_path.lower().endswith(('.svs', '.tif', '.tiff', '.ndpi')):
            return openslide.OpenSlide(slide_path)
        else:
            # Fallback to PIL for standard image formats
            return Image.open(slide_path)
    
    def extract_patches(
        self, 
        slide_path: str,
        return_coordinates: bool = True
    ) -> Tuple[List[np.ndarray], Optional[List[Tuple[int, int]]]]:
        """
        Extract patches from WSI.
        
        Args:
            slide_path: Path to the slide file
            return_coordinates: Whether to return patch coordinates
            
        Returns:
            Tuple of (patches, coordinates) where:
            - patches: List of patch arrays
            - coordinates: List of (x, y) coordinates (if return_coordinates=True)
        """
        slide = self._open_slide(slide_path)
        patches = []
        coordinates = [] if return_coordinates else None
        
        try:
            if hasattr(slide, 'level_dimensions'):
                # OpenSlide backend
                width, height = slide.level_dimensions[self.level]
                
                # Calculate grid dimensions
                cols = (width - self.patch_size) // self.stride + 1
                rows = (height - self.patch_size) // self.stride + 1
                
                total_patches = cols * rows
                progress_bar = tqdm(total=total_patches, desc="Extracting patches")
                
                for row in range(rows):
                    for col in range(cols):
                        x = col * self.stride
                        y = row * self.stride
                        
                        # Extract patch
                        patch = slide.read_region(
                            (x, y), self.level, (self.patch_size, self.patch_size)
                        )
                        patch = np.array(patch.convert('RGB'))
                        
                        # Skip background patches
                        if not self._is_background(patch):
                            patches.append(patch)
                            if return_coordinates:
                                coordinates.append((x, y))
                            
                            # Save patch if requested
                            if self.save_patches:
                                patch_name = f"patch_{row}_{col}.png"
                                patch_path = os.path.join(self.save_dir, patch_name)
                                Image.fromarray(patch).save(patch_path)
                        
                        progress_bar.update(1)
                
                progress_bar.close()
            
            else:
                # PIL backend for standard images
                if isinstance(slide, Image.Image):
                    slide_array = np.array(slide)
                else:
                    slide_array = slide
                
                height, width = slide_array.shape[:2]
                
                cols = (width - self.patch_size) // self.stride + 1
                rows = (height - self.patch_size) // self.stride + 1
                
                total_patches = cols * rows
                progress_bar = tqdm(total=total_patches, desc="Extracting patches")
                
                for row in range(rows):
                    for col in range(cols):
                        x = col * self.stride
                        y = row * self.stride
                        
                        patch = slide_array[y:y+self.patch_size, x:x+self.patch_size]
                        
                        if patch.shape[:2] == (self.patch_size, self.patch_size):
                            if not self._is_background(patch):
                                patches.append(patch)
                                if return_coordinates:
                                    coordinates.append((x, y))
                        
                        progress_bar.update(1)
                
                progress_bar.close()
        
        finally:
            if hasattr(slide, 'close'):
                slide.close()
        
        print(f"Extracted {len(patches)} patches from {slide_path}")
        return patches, coordinates
    
    def save_patches_to_h5(
        self, 
        slide_path: str, 
        output_path: str,
        compression: str = 'gzip'
    ) -> None:
        """
        Extract patches and save to HDF5 file for efficient storage.
        
        Args:
            slide_path: Path to the slide file
            output_path: Path to output HDF5 file
            compression: Compression method for HDF5
        """
        patches, coordinates = self.extract_patches(slide_path, return_coordinates=True)
        
        if not patches:
            print(f"No patches extracted from {slide_path}")
            return
        
        with h5py.File(output_path, 'w') as f:
            # Save patches
            patches_array = np.stack(patches)
            f.create_dataset(
                'patches', 
                data=patches_array, 
                compression=compression,
                compression_opts=9
            )
            
            # Save coordinates
            if coordinates:
                coords_array = np.array(coordinates)
                f.create_dataset(
                    'coordinates', 
                    data=coords_array, 
                    compression=compression
                )
            
            # Save metadata
            f.attrs['patch_size'] = self.patch_size
            f.attrs['stride'] = self.stride
            f.attrs['level'] = self.level
            f.attrs['num_patches'] = len(patches)
            f.attrs['slide_path'] = slide_path
        
        print(f"Saved {len(patches)} patches to {output_path}")


def preprocess_dataset(
    slide_paths: List[str],
    output_dir: str,
    patch_size: int = 256,
    stride: Optional[int] = None,
    level: int = 0,
    background_threshold: float = 0.7
) -> None:
    """
    Preprocess a dataset of WSI slides into patches.
    
    Args:
        slide_paths: List of paths to slide files
        output_dir: Directory to save processed patches
        patch_size: Size of extracted patches
        stride: Stride for patch extraction
        level: Magnification level
        background_threshold: Background detection threshold
    """
    os.makedirs(output_dir, exist_ok=True)
    
    tiler = WSITiler(
        patch_size=patch_size,
        stride=stride,
        level=level,
        background_threshold=background_threshold
    )
    
    for slide_path in tqdm(slide_paths, desc="Processing slides"):
        slide_name = os.path.splitext(os.path.basename(slide_path))[0]
        output_path = os.path.join(output_dir, f"{slide_name}_patches.h5")
        
        try:
            tiler.save_patches_to_h5(slide_path, output_path)
        except Exception as e:
            print(f"Error processing {slide_path}: {str(e)}")


def prepare_dataset_from_wsi(
    wsi_dir: str,
    labels_file: str,
    output_dir: str,
    patch_size: int = 256,
    stride: Optional[int] = None,
    level: int = 0,
    background_threshold: float = 0.7,
    file_extensions: List[str] = None
) -> None:
    """
    Prepare dataset from WSI directory.

    Args:
        wsi_dir: Directory containing WSI files
        labels_file: CSV file with labels
        output_dir: Output directory for patches
        patch_size: Patch size
        stride: Stride for extraction
        level: Magnification level
        background_threshold: Background threshold
        file_extensions: List of file extensions to process
    """
    if file_extensions is None:
        file_extensions = ['.svs', '.tif', '.tiff', '.ndpi']

    # Get all slide files
    import pandas as pd
    import glob

    slide_files = []
    for ext in file_extensions:
        slide_files.extend(glob.glob(os.path.join(wsi_dir, f'*{ext}')))

    if not slide_files:
        print(f"No slide files found in {wsi_dir}")
        return

    print(f"Found {len(slide_files)} slide files")

    # Process dataset
    preprocess_dataset(
        slide_paths=slide_files,
        output_dir=output_dir,
        patch_size=patch_size,
        stride=stride,
        level=level,
        background_threshold=background_threshold
    )


def create_label_file_template(
    output_path: str,
    slide_ids: List[str],
    labels: Optional[List[int]] = None
) -> None:
    """
    Create a template labels file.

    Args:
        output_path: Path to save CSV file
        slide_ids: List of slide IDs
        labels: Optional list of labels (defaults to 0)
    """
    import pandas as pd

    if labels is None:
        labels = [0] * len(slide_ids)

    df = pd.DataFrame({
        'slide_id': slide_ids,
        'label': labels
    })

    df.to_csv(output_path, index=False)
    print(f"Template labels file created at {output_path}")
    print(f"Contains {len(slide_ids)} slides")
    print("Please update the labels in this file with the correct values.")