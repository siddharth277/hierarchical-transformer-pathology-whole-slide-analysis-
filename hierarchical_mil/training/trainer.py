"""
Training utilities for hierarchical MIL transformer.
Includes trainer class, loss functions, and evaluation metrics.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from tqdm import tqdm
import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from ..models.hierarchical_mil import HierarchicalMILTransformer


class MILTrainer:
    """
    Trainer class for hierarchical MIL transformer.
    """
    
    def __init__(
        self,
        model: HierarchicalMILTransformer,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        test_loader: Optional[torch.utils.data.DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        criterion: Optional[nn.Module] = None,
        device: str = 'cuda',
        output_dir: str = 'outputs',
        log_interval: int = 10,
        save_interval: int = 1,
        max_epochs: int = 100,
        early_stopping_patience: int = 10,
        use_tensorboard: bool = True,
        gradient_clip_val: Optional[float] = None,
        accumulation_steps: int = 1
    ):
        """
        Initialize MIL trainer.
        
        Args:
            model: Hierarchical MIL transformer model
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            optimizer: Optimizer (Adam if None)
            scheduler: Learning rate scheduler
            criterion: Loss function (CrossEntropyLoss if None)
            device: Device to use for training
            output_dir: Directory to save outputs
            log_interval: Logging interval (epochs)
            save_interval: Model saving interval (epochs)
            max_epochs: Maximum training epochs
            early_stopping_patience: Early stopping patience
            use_tensorboard: Whether to use tensorboard logging
            gradient_clip_val: Gradient clipping value
            accumulation_steps: Gradient accumulation steps
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.output_dir = output_dir
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        self.gradient_clip_val = gradient_clip_val
        self.accumulation_steps = accumulation_steps
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize optimizer
        if optimizer is None:
            self.optimizer = optim.Adam(
                model.parameters(),
                lr=1e-4,
                weight_decay=1e-5
            )
        else:
            self.optimizer = optimizer
        
        # Initialize scheduler
        self.scheduler = scheduler
        
        # Initialize loss function
        if criterion is None:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = criterion
        
        # Initialize tensorboard
        if use_tensorboard:
            self.writer = SummaryWriter(os.path.join(output_dir, 'logs'))
        else:
            self.writer = None
        
        # Training state
        self.current_epoch = 0
        self.best_val_score = 0.0
        self.best_model_path = None
        self.early_stopping_counter = 0
        self.training_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch + 1}/{self.max_epochs}')
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device
            patches = batch['patches'].to(self.device)
            region_masks = batch['region_masks'].to(self.device)
            slide_masks = batch['slide_masks'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass
            outputs = self.model(
                patches=patches,
                region_masks=region_masks,
                slide_masks=slide_masks,
                return_attention=False,
                return_embeddings=False
            )
            
            logits = outputs['logits']
            loss = self.criterion(logits, labels)
            
            # Scale loss for gradient accumulation
            loss = loss / self.accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.accumulation_steps == 0:
                # Gradient clipping
                if self.gradient_clip_val is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
                
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            # Statistics
            total_loss += loss.item() * self.accumulation_steps
            predictions = logits.argmax(dim=1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)
            
            # Update progress bar
            current_acc = total_correct / total_samples
            progress_bar.set_postfix({
                'Loss': f'{total_loss / (batch_idx + 1):.4f}',
                'Acc': f'{current_acc:.4f}'
            })
        
        # Final gradient step if needed
        if len(self.train_loader) % self.accumulation_steps != 0:
            if self.gradient_clip_val is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        avg_loss = total_loss / len(self.train_loader)
        avg_acc = total_correct / total_samples
        
        return {'loss': avg_loss, 'accuracy': avg_acc}
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                # Move data to device
                patches = batch['patches'].to(self.device)
                region_masks = batch['region_masks'].to(self.device)
                slide_masks = batch['slide_masks'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    patches=patches,
                    region_masks=region_masks,
                    slide_masks=slide_masks,
                    return_attention=False,
                    return_embeddings=False
                )
                
                logits = outputs['logits']
                probabilities = outputs['probabilities']
                loss = self.criterion(logits, labels)
                
                # Statistics
                total_loss += loss.item()
                predictions = logits.argmax(dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy
        }
        
        # Add additional metrics for binary classification
        if self.model.num_classes == 2:
            try:
                auc = roc_auc_score(all_labels, np.array(all_probabilities)[:, 1])
                metrics['auc'] = auc
            except ValueError:
                metrics['auc'] = 0.0
        
        # Add precision, recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted', zero_division=0
        )
        metrics.update({
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
        
        return metrics
    
    def test_model(self, model_path: Optional[str] = None) -> Dict[str, float]:
        """Test the model."""
        if self.test_loader is None:
            return {}
        
        # Load model if path provided
        if model_path is not None:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        all_slide_ids = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc='Testing'):
                # Move data to device
                patches = batch['patches'].to(self.device)
                region_masks = batch['region_masks'].to(self.device)
                slide_masks = batch['slide_masks'].to(self.device)
                labels = batch['label'].to(self.device)
                slide_ids = batch['slide_id']
                
                # Forward pass
                outputs = self.model(
                    patches=patches,
                    region_masks=region_masks,
                    slide_masks=slide_masks,
                    return_attention=False,
                    return_embeddings=False
                )
                
                logits = outputs['logits']
                probabilities = outputs['probabilities']
                predictions = logits.argmax(dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_slide_ids.extend(slide_ids)
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted', zero_division=0
        )
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        # Add AUC for binary classification
        if self.model.num_classes == 2:
            try:
                auc = roc_auc_score(all_labels, np.array(all_probabilities)[:, 1])
                metrics['auc'] = auc
            except ValueError:
                metrics['auc'] = 0.0
        
        # Save detailed results
        results = {
            'slide_ids': all_slide_ids,
            'true_labels': all_labels,
            'predictions': all_predictions,
            'probabilities': all_probabilities
        }
        
        results_path = os.path.join(self.output_dir, 'test_results.json')
        with open(results_path, 'w') as f:
            json.dump({k: v if isinstance(v, list) else v.tolist() 
                      for k, v in results.items()}, f, indent=2)
        
        # Plot confusion matrix
        self._plot_confusion_matrix(all_labels, all_predictions)
        
        return metrics
    
    def _plot_confusion_matrix(self, true_labels: List[int], predictions: List[int]) -> None:
        """Plot and save confusion matrix."""
        cm = confusion_matrix(true_labels, predictions)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        cm_path = os.path.join(self.output_dir, 'confusion_matrix.png')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_model(self, epoch: int, metrics: Dict[str, float], is_best: bool = False) -> str:
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'training_history': self.training_history
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.output_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.output_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            self.best_model_path = best_path
        
        return checkpoint_path
    
    def train(self) -> None:
        """Main training loop."""
        print(f"Starting training for {self.max_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Output directory: {self.output_dir}")
        
        for epoch in range(self.max_epochs):
            self.current_epoch = epoch
            start_time = time.time()
            
            # Training
            train_metrics = self.train_epoch()
            
            # Validation
            val_metrics = self.validate_epoch()
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get('loss', train_metrics['loss']))
                else:
                    self.scheduler.step()
            
            # Update training history
            self.training_history['train_loss'].append(train_metrics['loss'])
            self.training_history['train_acc'].append(train_metrics['accuracy'])
            if val_metrics:
                self.training_history['val_loss'].append(val_metrics['loss'])
                self.training_history['val_acc'].append(val_metrics['accuracy'])
            self.training_history['learning_rate'].append(
                self.optimizer.param_groups[0]['lr']
            )
            
            # Tensorboard logging
            if self.writer is not None:
                self.writer.add_scalar('Train/Loss', train_metrics['loss'], epoch)
                self.writer.add_scalar('Train/Accuracy', train_metrics['accuracy'], epoch)
                self.writer.add_scalar('Train/LearningRate', 
                                     self.optimizer.param_groups[0]['lr'], epoch)
                
                if val_metrics:
                    self.writer.add_scalar('Val/Loss', val_metrics['loss'], epoch)
                    self.writer.add_scalar('Val/Accuracy', val_metrics['accuracy'], epoch)
                    if 'auc' in val_metrics:
                        self.writer.add_scalar('Val/AUC', val_metrics['auc'], epoch)
            
            # Check for best model
            current_score = val_metrics.get('accuracy', train_metrics['accuracy'])
            is_best = current_score > self.best_val_score
            if is_best:
                self.best_val_score = current_score
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1
            
            # Save model
            if (epoch + 1) % self.save_interval == 0 or is_best:
                self.save_model(epoch, {**train_metrics, **val_metrics}, is_best)
            
            # Logging
            epoch_time = time.time() - start_time
            if (epoch + 1) % self.log_interval == 0:
                log_str = f"Epoch {epoch + 1}/{self.max_epochs} ({epoch_time:.2f}s) - "
                log_str += f"Train Loss: {train_metrics['loss']:.4f}, "
                log_str += f"Train Acc: {train_metrics['accuracy']:.4f}"
                
                if val_metrics:
                    log_str += f", Val Loss: {val_metrics['loss']:.4f}, "
                    log_str += f"Val Acc: {val_metrics['accuracy']:.4f}"
                    if 'auc' in val_metrics:
                        log_str += f", Val AUC: {val_metrics['auc']:.4f}"
                
                log_str += f", LR: {self.optimizer.param_groups[0]['lr']:.6f}"
                print(log_str)
            
            # Early stopping
            if self.early_stopping_patience > 0 and self.early_stopping_counter >= self.early_stopping_patience:
                print(f"Early stopping triggered after {self.early_stopping_patience} epochs without improvement")
                break
        
        # Final model save
        self.save_model(self.current_epoch, {**train_metrics, **val_metrics}, False)
        
        # Close tensorboard
        if self.writer is not None:
            self.writer.close()
        
        print("Training completed!")
        print(f"Best validation score: {self.best_val_score:.4f}")
        if self.best_model_path:
            print(f"Best model saved at: {self.best_model_path}")


def create_optimizer(
    model: nn.Module,
    optimizer_type: str = 'adam',
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
    **kwargs
) -> torch.optim.Optimizer:
    """Create optimizer for model."""
    if optimizer_type.lower() == 'adam':
        return optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, **kwargs)
    elif optimizer_type.lower() == 'adamw':
        return optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, **kwargs)
    elif optimizer_type.lower() == 'sgd':
        return optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, 
                        momentum=kwargs.get('momentum', 0.9), **kwargs)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_type}")


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = 'plateau',
    **kwargs
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """Create learning rate scheduler."""
    if scheduler_type.lower() == 'plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=kwargs.get('patience', 5),
            factor=kwargs.get('factor', 0.5)
        )
    elif scheduler_type.lower() == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=kwargs.get('T_max', 100)
        )
    elif scheduler_type.lower() == 'step':
        return optim.lr_scheduler.StepLR(
            optimizer, step_size=kwargs.get('step_size', 30),
            gamma=kwargs.get('gamma', 0.1)
        )
    elif scheduler_type.lower() == 'none':
        return None
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_type}")