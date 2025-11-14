"""
Feature extraction and embedding generation utilities.
Supports extracting and saving features from trained models.
"""

import torch
import torch.nn as nn
import numpy as np
import h5py
import os
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import json


class FeatureExtractor:
    """
    Extract features from hierarchical MIL model at different levels.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        batch_size: int = 1,
        extract_patch_features: bool = True,
        extract_region_features: bool = True,
        extract_slide_features: bool = True
    ):
        """
        Initialize feature extractor.

        Args:
            model: Hierarchical MIL model
            device: Device to use
            batch_size: Batch size for extraction
            extract_patch_features: Whether to extract patch-level features
            extract_region_features: Whether to extract region-level features
            extract_slide_features: Whether to extract slide-level features
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.batch_size = batch_size
        self.extract_patch_features = extract_patch_features
        self.extract_region_features = extract_region_features
        self.extract_slide_features = extract_slide_features

    @torch.no_grad()
    def extract_features_from_loader(
        self,
        data_loader: torch.utils.data.DataLoader,
        save_dir: str,
        return_attention: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Extract features from a data loader.

        Args:
            data_loader: Data loader
            save_dir: Directory to save features
            return_attention: Whether to extract attention weights

        Returns:
            Dictionary of extracted features
        """
        os.makedirs(save_dir, exist_ok=True)

        all_features = {
            'slide_ids': [],
            'labels': []
        }

        if self.extract_patch_features:
            all_features['patch_features'] = []
        if self.extract_region_features:
            all_features['region_features'] = []
        if self.extract_slide_features:
            all_features['slide_features'] = []
        if return_attention:
            all_features['slide_attention'] = []
            all_features['region_attention'] = []

        for batch in tqdm(data_loader, desc='Extracting features'):
            # Move to device
            patches = batch['patches'].to(self.device)
            region_masks = batch['region_masks'].to(self.device)
            slide_masks = batch['slide_masks'].to(self.device)
            labels = batch['label']
            slide_ids = batch['slide_id']

            # Forward pass
            outputs = self.model(
                patches=patches,
                region_masks=region_masks,
                slide_masks=slide_masks,
                return_attention=return_attention,
                return_embeddings=True
            )

            # Store features
            all_features['slide_ids'].extend(slide_ids)
            all_features['labels'].extend(labels.cpu().numpy())

            if self.extract_patch_features:
                all_features['patch_features'].append(
                    outputs['patch_embeddings'].cpu().numpy()
                )

            if self.extract_region_features:
                all_features['region_features'].append(
                    outputs['region_embeddings'].cpu().numpy()
                )

            if self.extract_slide_features:
                all_features['slide_features'].append(
                    outputs['slide_embeddings'].cpu().numpy()
                )

            if return_attention:
                if 'slide_attention' in outputs:
                    all_features['slide_attention'].append(
                        outputs['slide_attention'].cpu().numpy()
                    )
                if 'region_attention' in outputs:
                    all_features['region_attention'].append(
                        outputs['region_attention'].cpu().numpy()
                    )

        # Concatenate features
        if self.extract_patch_features:
            all_features['patch_features'] = np.concatenate(all_features['patch_features'], axis=0)
        if self.extract_region_features:
            all_features['region_features'] = np.concatenate(all_features['region_features'], axis=0)
        if self.extract_slide_features:
            all_features['slide_features'] = np.concatenate(all_features['slide_features'], axis=0)
        if return_attention and len(all_features.get('slide_attention', [])) > 0:
            all_features['slide_attention'] = np.concatenate(all_features['slide_attention'], axis=0)
        if return_attention and len(all_features.get('region_attention', [])) > 0:
            all_features['region_attention'] = np.concatenate(all_features['region_attention'], axis=0)

        all_features['labels'] = np.array(all_features['labels'])

        # Save features
        self.save_features(all_features, save_dir)

        return all_features

    def save_features(self, features: Dict[str, np.ndarray], save_dir: str) -> None:
        """
        Save extracted features to disk.

        Args:
            features: Dictionary of features
            save_dir: Directory to save features
        """
        # Save as HDF5
        h5_path = os.path.join(save_dir, 'features.h5')
        with h5py.File(h5_path, 'w') as f:
            for key, value in features.items():
                if isinstance(value, np.ndarray):
                    f.create_dataset(key, data=value, compression='gzip')
                elif isinstance(value, list) and len(value) > 0:
                    # Handle list of strings (slide_ids)
                    if isinstance(value[0], str):
                        dt = h5py.special_dtype(vlen=str)
                        f.create_dataset(key, data=value, dtype=dt)

        # Save metadata
        metadata = {
            'num_samples': len(features['slide_ids']),
            'feature_keys': list(features.keys()),
            'slide_ids': features['slide_ids'] if isinstance(features['slide_ids'], list) else features['slide_ids'].tolist()
        }

        if self.extract_slide_features and 'slide_features' in features:
            metadata['slide_feature_dim'] = features['slide_features'].shape[-1]
        if self.extract_region_features and 'region_features' in features:
            metadata['region_feature_dim'] = features['region_features'].shape[-1]
        if self.extract_patch_features and 'patch_features' in features:
            metadata['patch_feature_dim'] = features['patch_features'].shape[-1]

        json_path = os.path.join(save_dir, 'features_metadata.json')
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Features saved to {save_dir}")

    @staticmethod
    def load_features(feature_dir: str) -> Dict[str, np.ndarray]:
        """
        Load saved features.

        Args:
            feature_dir: Directory containing saved features

        Returns:
            Dictionary of features
        """
        h5_path = os.path.join(feature_dir, 'features.h5')
        features = {}

        with h5py.File(h5_path, 'r') as f:
            for key in f.keys():
                features[key] = f[key][:]

        return features


class EmbeddingDatabase:
    """
    Database for storing and retrieving slide embeddings.
    Useful for similarity search and retrieval.
    """

    def __init__(self, embedding_dim: int):
        """
        Initialize embedding database.

        Args:
            embedding_dim: Dimension of embeddings
        """
        self.embedding_dim = embedding_dim
        self.embeddings = []
        self.slide_ids = []
        self.labels = []
        self.metadata = []

    def add_embedding(
        self,
        embedding: np.ndarray,
        slide_id: str,
        label: int,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Add an embedding to the database.

        Args:
            embedding: Slide embedding
            slide_id: Slide identifier
            label: Slide label
            metadata: Optional metadata dict
        """
        assert embedding.shape[-1] == self.embedding_dim, "Embedding dimension mismatch"

        self.embeddings.append(embedding)
        self.slide_ids.append(slide_id)
        self.labels.append(label)
        self.metadata.append(metadata or {})

    def build_index(self) -> None:
        """Build index for efficient similarity search."""
        self.embeddings_matrix = np.array(self.embeddings)

        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(self.embeddings_matrix, axis=1, keepdims=True)
        self.normalized_embeddings = self.embeddings_matrix / (norms + 1e-10)

    def find_similar(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        return_distances: bool = True
    ) -> Tuple[List[str], Optional[np.ndarray]]:
        """
        Find most similar slides to query embedding.

        Args:
            query_embedding: Query embedding
            top_k: Number of similar slides to return
            return_distances: Whether to return similarity scores

        Returns:
            List of similar slide IDs and optionally distances
        """
        # Normalize query
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)

        # Compute cosine similarity
        similarities = np.dot(self.normalized_embeddings, query_norm)

        # Get top-k
        top_idx = np.argsort(similarities)[-top_k:][::-1]
        similar_ids = [self.slide_ids[i] for i in top_idx]

        if return_distances:
            distances = similarities[top_idx]
            return similar_ids, distances
        else:
            return similar_ids, None

    def find_similar_by_label(
        self,
        query_embedding: np.ndarray,
        target_label: int,
        top_k: int = 10
    ) -> Tuple[List[str], np.ndarray]:
        """
        Find similar slides with a specific label.

        Args:
            query_embedding: Query embedding
            target_label: Target label to match
            top_k: Number of results

        Returns:
            List of slide IDs and similarity scores
        """
        # Filter by label
        label_mask = np.array(self.labels) == target_label
        if not label_mask.any():
            return [], np.array([])

        # Normalize query
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)

        # Compute similarities for filtered slides
        filtered_embeddings = self.normalized_embeddings[label_mask]
        similarities = np.dot(filtered_embeddings, query_norm)

        # Get top-k
        top_k = min(top_k, len(similarities))
        top_idx = np.argsort(similarities)[-top_k:][::-1]

        # Map back to original indices
        original_indices = np.where(label_mask)[0][top_idx]
        similar_ids = [self.slide_ids[i] for i in original_indices]
        distances = similarities[top_idx]

        return similar_ids, distances

    def save(self, save_path: str) -> None:
        """
        Save embedding database.

        Args:
            save_path: Path to save database
        """
        data = {
            'embeddings': self.embeddings_matrix,
            'slide_ids': self.slide_ids,
            'labels': self.labels,
            'metadata': self.metadata,
            'embedding_dim': self.embedding_dim
        }

        np.savez(save_path, **{k: v for k, v in data.items() if isinstance(v, np.ndarray)})

        # Save non-numpy data as JSON
        json_path = save_path.replace('.npz', '_meta.json')
        with open(json_path, 'w') as f:
            json.dump({
                'slide_ids': self.slide_ids,
                'labels': self.labels,
                'metadata': self.metadata,
                'embedding_dim': self.embedding_dim
            }, f, indent=2)

    @classmethod
    def load(cls, save_path: str) -> 'EmbeddingDatabase':
        """
        Load embedding database.

        Args:
            save_path: Path to load from

        Returns:
            Loaded database
        """
        # Load numpy data
        data = np.load(save_path)

        # Load JSON metadata
        json_path = save_path.replace('.npz', '_meta.json')
        with open(json_path, 'r') as f:
            metadata = json.load(f)

        db = cls(embedding_dim=metadata['embedding_dim'])
        db.embeddings_matrix = data['embeddings']
        db.slide_ids = metadata['slide_ids']
        db.labels = metadata['labels']
        db.metadata = metadata['metadata']
        db.build_index()

        return db


def extract_and_save_features(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    save_dir: str,
    device: str = 'cuda',
    return_attention: bool = True
) -> None:
    """
    Convenience function to extract and save features.

    Args:
        model: Trained model
        data_loader: Data loader
        save_dir: Directory to save features
        device: Device to use
        return_attention: Whether to extract attention
    """
    extractor = FeatureExtractor(
        model=model,
        device=device,
        extract_patch_features=True,
        extract_region_features=True,
        extract_slide_features=True
    )

    features = extractor.extract_features_from_loader(
        data_loader=data_loader,
        save_dir=save_dir,
        return_attention=return_attention
    )

    print(f"Extracted features for {len(features['slide_ids'])} slides")
    if 'slide_features' in features:
        print(f"Slide features shape: {features['slide_features'].shape}")
    if 'region_features' in features:
        print(f"Region features shape: {features['region_features'].shape}")
    if 'patch_features' in features:
        print(f"Patch features shape: {features['patch_features'].shape}")
