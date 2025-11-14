"""
Unit tests for hierarchical MIL transformer components.
"""

import unittest
import torch
import numpy as np
import tempfile
import os
import shutil
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from hierarchical_mil.models.patch_encoder import PatchEncoder
from hierarchical_mil.models.attention import AttentionAggregator, PatchToRegionAggregator, RegionToSlideAggregator
from hierarchical_mil.models.hierarchical_mil import HierarchicalMILTransformer
from hierarchical_mil.data.wsi_preprocessing import WSITiler


class TestPatchEncoder(unittest.TestCase):
    """Test patch encoder functionality."""
    
    def setUp(self):
        self.batch_size = 2
        self.patch_size = 224
        self.feature_dim = 256
    
    def test_resnet_encoder(self):
        """Test ResNet-based patch encoder."""
        encoder = PatchEncoder(
            backbone_type='resnet18',  # Use smaller model for testing
            feature_dim=self.feature_dim,
            pretrained=False,
            patch_size=self.patch_size
        )
        
        # Test input
        patches = torch.randn(self.batch_size, 3, self.patch_size, self.patch_size)
        features = encoder(patches)
        
        self.assertEqual(features.shape, (self.batch_size, self.feature_dim))
        self.assertFalse(torch.isnan(features).any())
    
    def test_vit_encoder(self):
        """Test ViT-based patch encoder."""
        encoder = PatchEncoder(
            backbone_type='vit_base',
            feature_dim=self.feature_dim,
            pretrained=False,
            patch_size=224  # ViT requires specific size
        )
        
        # Test input
        patches = torch.randn(self.batch_size, 3, 224, 224)
        features = encoder(patches)
        
        self.assertEqual(features.shape, (self.batch_size, self.feature_dim))
        self.assertFalse(torch.isnan(features).any())
    
    def test_batch_processing(self):
        """Test batch processing with multiple patches."""
        encoder = PatchEncoder(
            backbone_type='resnet18',
            feature_dim=self.feature_dim,
            pretrained=False,
            patch_size=self.patch_size
        )
        
        # Test batch of patches
        batch_size, num_patches = 2, 5
        patches = torch.randn(batch_size, num_patches, 3, self.patch_size, self.patch_size)
        features = encoder(patches)
        
        self.assertEqual(features.shape, (batch_size, num_patches, self.feature_dim))
        self.assertFalse(torch.isnan(features).any())


class TestAttentionAggregator(unittest.TestCase):
    """Test attention aggregation modules."""
    
    def setUp(self):
        self.batch_size = 2
        self.seq_len = 10
        self.input_dim = 256
        self.hidden_dim = 128
    
    def test_attention_aggregator(self):
        """Test basic attention aggregator."""
        aggregator = AttentionAggregator(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_heads=4,
            num_layers=2
        )
        
        # Test input
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        output, attention = aggregator(x, return_attention=True)
        
        self.assertEqual(output.shape, (self.batch_size, self.hidden_dim))
        self.assertIsNotNone(attention)
        self.assertFalse(torch.isnan(output).any())
    
    def test_patch_to_region_aggregator(self):
        """Test patch-to-region aggregator."""
        aggregator = PatchToRegionAggregator(
            patch_dim=self.input_dim,
            region_dim=self.hidden_dim,
            num_heads=4,
            num_layers=2
        )
        
        # Test input
        patches = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        region_features, attention = aggregator(patches, return_attention=True)
        
        self.assertEqual(region_features.shape, (self.batch_size, self.hidden_dim))
        self.assertFalse(torch.isnan(region_features).any())
    
    def test_region_to_slide_aggregator(self):
        """Test region-to-slide aggregator."""
        aggregator = RegionToSlideAggregator(
            region_dim=self.input_dim,
            slide_dim=self.hidden_dim,
            num_heads=4,
            num_layers=2
        )
        
        # Test input
        regions = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        slide_features, attention = aggregator(regions, return_attention=True)
        
        self.assertEqual(slide_features.shape, (self.batch_size, self.hidden_dim))
        self.assertFalse(torch.isnan(slide_features).any())
    
    def test_masking(self):
        """Test attention with masking."""
        aggregator = AttentionAggregator(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_heads=4,
            num_layers=1
        )
        
        # Create input and mask
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        mask = torch.ones(self.batch_size, self.seq_len, dtype=torch.bool)
        mask[:, self.seq_len//2:] = False  # Mask half the sequence
        
        output, attention = aggregator(x, mask=mask, return_attention=True)
        
        self.assertEqual(output.shape, (self.batch_size, self.hidden_dim))
        self.assertFalse(torch.isnan(output).any())


class TestHierarchicalMILTransformer(unittest.TestCase):
    """Test the complete hierarchical MIL transformer."""
    
    def setUp(self):
        self.batch_size = 1  # Small batch for testing
        self.regions_per_slide = 4
        self.patches_per_region = 8
        self.patch_size = 224
        self.num_classes = 2
    
    def test_model_forward(self):
        """Test forward pass through complete model."""
        model = HierarchicalMILTransformer(
            backbone_type='resnet18',  # Use smaller model for testing
            patch_feature_dim=128,
            pretrained_backbone=False,
            region_dim=128,
            patches_per_region=self.patches_per_region,
            region_attention_heads=4,
            region_attention_layers=1,
            slide_dim=128,
            regions_per_slide=self.regions_per_slide,
            slide_attention_heads=4,
            slide_attention_layers=1,
            num_classes=self.num_classes,
            dropout=0.1
        )
        
        # Create test input
        patches = torch.randn(
            self.batch_size, 
            self.regions_per_slide, 
            self.patches_per_region,
            3, 
            self.patch_size, 
            self.patch_size
        )
        
        # Create masks
        region_masks = torch.ones(self.batch_size, self.regions_per_slide, self.patches_per_region, dtype=torch.bool)
        slide_masks = torch.ones(self.batch_size, self.regions_per_slide, dtype=torch.bool)
        
        # Forward pass
        outputs = model(
            patches=patches,
            region_masks=region_masks,
            slide_masks=slide_masks,
            return_attention=True,
            return_embeddings=True
        )
        
        # Check outputs
        self.assertIn('logits', outputs)
        self.assertIn('probabilities', outputs)
        self.assertIn('slide_embeddings', outputs)
        self.assertIn('region_embeddings', outputs)
        self.assertIn('patch_embeddings', outputs)
        
        # Check shapes
        self.assertEqual(outputs['logits'].shape, (self.batch_size, self.num_classes))
        self.assertEqual(outputs['probabilities'].shape, (self.batch_size, self.num_classes))
        self.assertEqual(outputs['slide_embeddings'].shape, (self.batch_size, 128))
        
        # Check no NaN values
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                self.assertFalse(torch.isnan(value).any(), f"NaN found in {key}")
    
    def test_model_with_partial_masks(self):
        """Test model with partial masks (some regions/patches missing)."""
        model = HierarchicalMILTransformer(
            backbone_type='resnet18',
            patch_feature_dim=64,
            pretrained_backbone=False,
            region_dim=64,
            patches_per_region=self.patches_per_region,
            slide_dim=64,
            regions_per_slide=self.regions_per_slide,
            num_classes=self.num_classes
        )
        
        # Create test input
        patches = torch.randn(
            self.batch_size, 
            self.regions_per_slide, 
            self.patches_per_region,
            3, 
            self.patch_size, 
            self.patch_size
        )
        
        # Create partial masks
        region_masks = torch.ones(self.batch_size, self.regions_per_slide, self.patches_per_region, dtype=torch.bool)
        region_masks[:, :, self.patches_per_region//2:] = False  # Mask half the patches
        
        slide_masks = torch.ones(self.batch_size, self.regions_per_slide, dtype=torch.bool)
        slide_masks[:, self.regions_per_slide//2:] = False  # Mask half the regions
        
        # Forward pass
        outputs = model(
            patches=patches,
            region_masks=region_masks,
            slide_masks=slide_masks
        )
        
        # Check outputs
        self.assertEqual(outputs['logits'].shape, (self.batch_size, self.num_classes))
        self.assertFalse(torch.isnan(outputs['logits']).any())


class TestWSITiler(unittest.TestCase):
    """Test WSI tiling functionality."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a dummy image
        self.image_size = (1000, 1000, 3)
        self.dummy_image = np.random.randint(0, 255, self.image_size, dtype=np.uint8)
        self.image_path = os.path.join(self.temp_dir, 'test_image.png')
        
        from PIL import Image
        Image.fromarray(self.dummy_image).save(self.image_path)
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_patch_extraction(self):
        """Test basic patch extraction."""
        tiler = WSITiler(
            patch_size=256,
            stride=256,
            background_threshold=0.9  # High threshold to include most patches
        )
        
        patches, coordinates = tiler.extract_patches(self.image_path)
        
        # Should extract some patches
        self.assertGreater(len(patches), 0)
        self.assertEqual(len(patches), len(coordinates))
        
        # Check patch dimensions
        for patch in patches:
            self.assertEqual(patch.shape, (256, 256, 3))
    
    def test_h5_saving(self):
        """Test saving patches to H5 file."""
        tiler = WSITiler(patch_size=128, stride=128)
        
        output_path = os.path.join(self.temp_dir, 'patches.h5')
        tiler.save_patches_to_h5(self.image_path, output_path)
        
        # Check if file was created
        self.assertTrue(os.path.exists(output_path))
        
        # Check file contents
        import h5py
        with h5py.File(output_path, 'r') as f:
            self.assertIn('patches', f)
            patches = f['patches'][:]
            self.assertGreater(len(patches), 0)
            self.assertEqual(len(patches.shape), 4)  # [N, H, W, C]


def run_tests():
    """Run all tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_cases = [
        TestPatchEncoder,
        TestAttentionAggregator,
        TestHierarchicalMILTransformer,
        TestWSITiler
    ]
    
    for test_case in test_cases:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_case)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    if not success:
        sys.exit(1)