"""
Dataset and data loading utilities for hierarchical MIL.
Supports CAMELYON16, PANDA, TCGA, and custom pathology datasets.
"""

import os
import torch
import torch.utils.data as data
import numpy as np
import pandas as pd
import h5py
from typing import Dict, List, Optional, Tuple, Union, Callable
from PIL import Image
import json
import pickle
from tqdm import tqdm
import random

# from .wsi_preprocessing import WSITiler # Assuming this is in another file
os.add_dll_directory(r"C:\openslide\openslide-bin-4.0.0.8-windows-x64\bin")

class WSIDataset(data.Dataset):
    """
    Dataset for WSI patches with hierarchical structure.
    """
    
    def __init__(
        self,
        data_dir: str,
        labels_file: str,
        patches_per_region: int = 64,
        regions_per_slide: int = 32,
        patch_size: int = 256,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        preload_patches: bool = False,
        cache_patches: bool = True,
        augmentation_prob: float = 0.0
    ):
        self.data_dir = data_dir
        self.patches_per_region = patches_per_region
        self.regions_per_slide = regions_per_slide
        self.patch_size = patch_size
        self.transform = transform
        self.target_transform = target_transform
        self.preload_patches = preload_patches
        self.cache_patches = cache_patches
        self.augmentation_prob = augmentation_prob
        
        # Load labels
        self.labels_df = pd.read_csv(labels_file)
        self.slide_ids = self.labels_df['slide_id'].tolist()
        self.labels = self.labels_df['label'].tolist()
        
        
        # === FIX FOR LABEL RE-MAPPING ===
        # Check if labels are already integers (0, 1, etc.)
        if all(isinstance(label, (int, float, np.integer, np.floating)) for label in self.labels):
            # If they are, use them directly
            self.label_indices = [int(l) for l in self.labels]
            # Create a map that includes ALL possible classes (0 and 1)
            self.label_to_idx = {0: 0, 1: 1} 
        else:
            # If they are strings (e.g., 'Normal', 'Tumor'), create a map
            self.label_to_idx = {}
            unique_labels = sorted(list(set(self.labels)))
            for idx, label in enumerate(unique_labels):
                self.label_to_idx[label] = idx
            
            # Convert labels to indices
            self.label_indices = [self.label_to_idx[label] for label in self.labels]
        # ================================
        
        
        # Cache for patches
        self.patch_cache = {} if cache_patches else None
        
        # Preload patches if requested
        if preload_patches:
            self._preload_all_patches()
    
    def _preload_all_patches(self):
        """Preload all patches into memory."""
        print("Preloading all patches...")
        self.preloaded_patches = {}
        for i in tqdm(range(len(self.slide_ids))):
            slide_id = self.slide_ids[i]
            patches = self._load_slide_patches(slide_id)
            self.preloaded_patches[slide_id] = patches
    
    
    # === FIX FOR .NPY LOADING, PICKLING, KEYERROR, AND GRAYSCALE ===
    def _load_slide_patches(self, slide_id: str) -> np.ndarray:
        """
        Load patches for a slide from a .npy file.
        The slide_id is the relative path from the labels CSV.
        """
        patch_file = os.path.join(self.data_dir, slide_id)
        
        if not os.path.exists(patch_file):
            print(f"DEBUG: File not found at path: {patch_file}")
            raise FileNotFoundError(f"No patch file found for slide {slide_id} at {patch_file}")
        
        try:
            loaded_data = np.load(patch_file, allow_pickle=True)
            
            if loaded_data.shape == ():
                patches_dict = loaded_data.item()
            else:
                patches_dict = loaded_data # Failsafe

            # Get the array from the dictionary
            patches = patches_dict['image']

            return patches
            
        except Exception as e:
            print(f"Error loading .npy file {patch_file}: {e}")
            raise
    # ===============================================================


    def _organize_patches_hierarchically(self, patches: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Organize patches into hierarchical structure (regions and patches).
        """
        
        if not isinstance(patches, np.ndarray) or patches.size == 0:
            patches = np.array([])
            
        num_patches = len(patches)
        total_patches_needed = self.regions_per_slide * self.patches_per_region
        
        # Infer patch shape
        if num_patches > 0:
            # === FIX FOR GRAYSCALE IMAGES ===
            if patches[0].ndim == 2: # Shape is (H, W)
                patch_h, patch_w = patches[0].shape
                patch_c = 1 # Grayscale
            elif patches[0].ndim == 3: # Shape is (H, W, C)
                patch_h, patch_w, patch_c = patches[0].shape
            else:
                raise ValueError(f"Unexpected patch shape: {patches[0].shape}")
        else:
            patch_h, patch_w, patch_c = self.patch_size, self.patch_size, 3
            
        # Create a 1-channel or 3-channel array as needed
        organized_patches = np.zeros(
            (self.regions_per_slide, self.patches_per_region, patch_h, patch_w, patch_c),
            dtype=np.uint8
        )
        # ==================================
        
        region_masks = np.zeros((self.regions_per_slide, self.patches_per_region), dtype=bool)
        slide_masks = np.zeros(self.regions_per_slide, dtype=bool)
        
        if num_patches == 0:
            return organized_patches, region_masks, slide_masks
        
        if num_patches < total_patches_needed:
            indices = np.tile(np.arange(num_patches), 
                            (total_patches_needed // num_patches) + 1)[:total_patches_needed]
        else:
            indices = np.random.choice(num_patches, total_patches_needed, replace=False)
        
        patch_idx = 0
        for region_idx in range(self.regions_per_slide):
            patches_in_region = min(self.patches_per_region, 
                                  total_patches_needed - patch_idx)
            
            if patches_in_region > 0:
                slide_masks[region_idx] = True
                
                for patch_in_region_idx in range(patches_in_region):
                    organized_patches[region_idx, patch_in_region_idx] = patches[indices[patch_idx]]
                    region_masks[region_idx, patch_in_region_idx] = True
                    patch_idx += 1
        
        return organized_patches, region_masks, slide_masks
    
    def __len__(self) -> int:
        return len(self.slide_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        slide_id = self.slide_ids[idx]
        label = self.label_indices[idx]
        
        if self.preload_patches:
            patches = self.preloaded_patches[slide_id]
        elif self.cache_patches and slide_id in self.patch_cache:
            patches = self.patch_cache[slide_id]
        else:
            patches = self._load_slide_patches(slide_id)
            if self.cache_patches:
                self.patch_cache[slide_id] = patches
        
        # === FIX FOR GRAYSCALE IMAGES ===
        # Check if patches are grayscale (N, H, W) and convert to RGB (N, H, W, 3)
        if patches.ndim == 3:
            patches = np.repeat(patches[..., np.newaxis], 3, axis=-1)
        # ================================
        
        organized_patches, region_masks, slide_masks = self._organize_patches_hierarchically(patches)
        
        if self.transform is not None:
            transformed_patches = np.zeros_like(organized_patches, dtype=np.float32)
            for r in range(self.regions_per_slide):
                for p in range(self.patches_per_region):
                    if region_masks[r, p]:
                        patch = organized_patches[r, p]
                        if self.augmentation_prob > 0 and random.random() < self.augmentation_prob:
                            patch = self._apply_augmentation(patch)
                        
                        if not isinstance(patch, Image.Image):
                            patch = Image.fromarray(patch)
                            
                        transformed_patches[r, p] = self.transform(patch)
            organized_patches = transformed_patches
        
        patches_tensor = torch.from_numpy(organized_patches).float()
        
        if self.transform is None:
            patches_tensor = patches_tensor / 255.0
            
        # Always permute to [R, P, C, H, W]
        patches_tensor = patches_tensor.permute(0, 1, 4, 2, 3) 
        
        region_masks_tensor = torch.from_numpy(region_masks).bool()
        slide_masks_tensor = torch.from_numpy(slide_masks).bool()
        
        if self.target_transform is not None:
            label = self.target_transform(label)
        
        return {
            'patches': patches_tensor,
            'region_masks': region_masks_tensor,
            'slide_masks': slide_masks_tensor,
            'label': torch.tensor(label, dtype=torch.long),
            'slide_id': slide_id
        }
    
    def _apply_augmentation(self, patch: np.ndarray) -> np.ndarray:
        """Apply simple augmentations to a patch."""
        if random.random() < 0.5:
            k = random.randint(1, 3)
            patch = np.rot90(patch, k)
        if random.random() < 0.5:
            patch = np.fliplr(patch)
        if random.random() < 0.5:
            patch = np.flipud(patch)
        return np.ascontiguousarray(patch)

