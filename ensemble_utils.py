"""
Ensemble Model Utilities for Domain-Adaptation Multi-Cancer Framework

This module provides utilities for creating and evaluating ensemble models
using a "leave-one-out" domain adaptation approach where multiple domain-specific
models are combined to predict on a held-out target domain.

Key Features:
- Load multiple pre-trained models from different domains
- Create ensemble predictions using various aggregation strategies
- Leave-one-domain-out evaluation framework
- Performance metrics and visualization for ensemble results
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd

# Import transformers for HuggingFace models
try:
    from transformers import ResNetForImageClassification, ResNetConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers library not available. HuggingFace model loading will fail.")

# Add the src directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Import from existing codebase
try:
    from src.domain_adaptation_ct.learn.architectures import ARCHITECTURE_REGISTRY, ResNet50DANN, ResNet50Baseline
except ImportError:
    # Alternative import approach if the above fails
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
    from domain_adaptation_ct.learn.architectures import ARCHITECTURE_REGISTRY, ResNet50DANN, ResNet50Baseline

from cancer_utils import load_cancer_dataset, create_domain_datasets, get_standard_transforms
from torch.utils.data import DataLoader


class EnsembleModel:
    """
    Ensemble model that combines predictions from multiple domain-specific models.
    """
    
    def __init__(self, models: Dict[str, torch.nn.Module], ensemble_method: str = 'average'):
        """
        Initialize ensemble model.
        
        Args:
            models: Dictionary mapping domain names to trained models
            ensemble_method: Method for combining predictions ('average', 'weighted', 'voting')
        """
        self.models = models
        self.ensemble_method = ensemble_method
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move all models to device and set to eval mode
        for model in self.models.values():
            model.to(self.device)
            model.eval()
    
    def predict_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Make ensemble predictions on a batch of data.
        
        Args:
            batch: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Ensemble predictions of shape (batch_size, num_classes)
        """
        batch = batch.to(self.device)
        predictions = []
        
        with torch.no_grad():
            for domain_name, model in self.models.items():
                # Get model predictions
                outputs = model(batch)
                
                # Handle different model output formats
                if hasattr(outputs, 'branch1_logits') and outputs.branch1_logits is not None:
                    # DANN model with branched output
                    logits = outputs.branch1_logits
                elif hasattr(outputs, 'logits'):
                    # Standard model output
                    logits = outputs.logits
                elif isinstance(outputs, torch.Tensor):
                    # Direct tensor output
                    logits = outputs
                else:
                    # Try to extract logits from output object
                    logits = getattr(outputs, 'logits', outputs)
                
                # Apply softmax to get probabilities
                probs = F.softmax(logits, dim=-1)
                predictions.append(probs)
        
        # Combine predictions based on ensemble method
        if self.ensemble_method == 'average':
            ensemble_probs = torch.mean(torch.stack(predictions), dim=0)
        elif self.ensemble_method == 'voting':
            # Hard voting - take mode of predicted classes
            pred_classes = torch.stack([torch.argmax(p, dim=-1) for p in predictions])
            ensemble_classes = torch.mode(pred_classes, dim=0)[0]
            ensemble_probs = F.one_hot(ensemble_classes, num_classes=predictions[0].shape[-1]).float()
        else:
            raise ValueError(f"Unsupported ensemble method: {self.ensemble_method}")
        
        return ensemble_probs
    
    def predict_dataset(self, dataset, batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on an entire dataset.
        
        Args:
            dataset: PyTorch dataset to predict on
            batch_size: Batch size for prediction
            
        Returns:
            Tuple of (predictions, ground_truth_labels)
        """
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        all_predictions = []
        all_labels = []
        
        for batch in dataloader:
            # Extract pixel values and labels from batch
            if isinstance(batch, dict):
                pixel_values = batch['pixel_values']
                labels = batch.get('labels', batch.get('labels1', None))
            else:
                pixel_values, labels = batch
            
            # Get ensemble predictions
            ensemble_probs = self.predict_batch(pixel_values)
            predicted_classes = torch.argmax(ensemble_probs, dim=-1)
            
            all_predictions.extend(predicted_classes.cpu().numpy())
            if labels is not None:
                all_labels.extend(labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels)
        
        return np.array(all_predictions), np.array(all_labels) if all_labels else None


def load_model_from_checkpoint(model_path: str, model_type: str = 'ResNet50DANN') -> torch.nn.Module:
    """
    Load a pre-trained model from checkpoint.
    
    Args:
        model_path: Path to the model directory (should contain model.safetensors and config.json)
        model_type: Type of model architecture ('ResNet50DANN' or 'ResNet50Baseline')
        
    Returns:
        Loaded model
    """
    # Load model configuration
    config_path = os.path.join(model_path, 'config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        num_classes = config.get('num_classes', 2)
        model_type_from_config = config.get('model_type', 'resnet')
    else:
        num_classes = 2  # Default for binary classification
        model_type_from_config = 'resnet'
    
    # Load model weights
    model_weights_path = os.path.join(model_path, 'model.safetensors')
    
    # First, check what keys are actually in the state dict
    from safetensors.torch import load_file
    state_dict = load_file(model_weights_path)
    
    # Determine the actual model architecture based on keys present
    has_resnet_structure = any('resnet.' in key for key in state_dict.keys())
    has_classifier = any('classifier' in key for key in state_dict.keys())
    has_dann_structure = any('branch1' in key for key in state_dict.keys()) or any('branch2' in key for key in state_dict.keys())
    
    if has_resnet_structure and has_classifier and not has_dann_structure:
        # This is a HuggingFace Transformers ResNet model
        print(f"  Note: Detected HuggingFace ResNet architecture - creating compatible wrapper")
        
        # Create a simple wrapper model that matches the HuggingFace ResNet structure
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library is required but not available. Install with: pip install transformers")
        
        # Create configuration matching the saved model
        resnet_config = ResNetConfig(
            num_labels=num_classes,
            num_channels=3,
            embedding_size=64,
            hidden_sizes=[256, 512, 1024, 2048],
            depths=[3, 4, 6, 3],
            layer_type="bottleneck",
            hidden_act="relu",
            downsample_in_first_stage=False,
            downsample_in_bottleneck=False
        )
        
        # Create the model
        model = ResNetForImageClassification(resnet_config)
        
        # Load the state dict
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        
    elif model_type == 'ResNet50DANN' and has_dann_structure:
        # Load DANN model with default parameters
        model = ResNet50DANN.load(
            file_path=model_weights_path,
            num_classes=num_classes,
            lamb_initial=1.0,  # Default lambda value
            ld_scale=1.0       # Default ld_scale
        )
    elif model_type == 'ResNet50Baseline' and not has_resnet_structure:
        # Load custom baseline model (only if not HuggingFace structure)
        model = ResNet50Baseline(num_classes=num_classes)
        model.load_state_dict(state_dict)
        model.eval()
    else:
        raise ValueError(f"Unsupported model architecture or incompatible state dict. "
                        f"Found keys: {list(state_dict.keys())[:5]}...")
    
    return model


def load_all_domain_models(models_base_path: str, model_type: str = 'ResNet50DANN') -> Dict[str, torch.nn.Module]:
    """
    Load all domain-specific models from the models directory.
    
    Args:
        models_base_path: Base path containing domain model directories
        model_type: Type of model architecture
        
    Returns:
        Dictionary mapping domain names to loaded models
    """
    models = {}
    domain_mapping = {
        'final_breast_model': 'Breast Cancer',
        'final_colon_model': 'Colon Cancer', 
        'final_kidney_model': 'Kidney Cancer',
        'final_lung_model': 'Lung Cancer'
    }
    
    for model_dir, domain_name in domain_mapping.items():
        model_path = os.path.join(models_base_path, model_dir, 'final_model')
        
        if os.path.exists(model_path):
            print(f"Loading model for {domain_name} from {model_path}")
            try:
                model = load_model_from_checkpoint(model_path, model_type)
                models[domain_name] = model
                print(f"✓ Successfully loaded {domain_name} model")
            except Exception as e:
                print(f"✗ Failed to load {domain_name} model: {e}")
        else:
            print(f"✗ Model path not found: {model_path}")
    
    return models


def create_leave_one_out_ensembles(models: Dict[str, torch.nn.Module]) -> Dict[str, EnsembleModel]:
    """
    Create leave-one-out ensemble models.
    
    Args:
        models: Dictionary of all domain models
        
    Returns:
        Dictionary mapping target domains to ensemble models (excluding that domain)
    """
    ensembles = {}
    
    for target_domain in models.keys():
        # Create ensemble excluding the target domain
        ensemble_models = {domain: model for domain, model in models.items() 
                          if domain != target_domain}
        
        if len(ensemble_models) > 0:
            ensemble = EnsembleModel(ensemble_models, ensemble_method='average')
            ensembles[target_domain] = ensemble
            print(f"Created ensemble for testing on '{target_domain}' using models: {list(ensemble_models.keys())}")
    
    return ensembles


def evaluate_ensemble_performance(ensemble: EnsembleModel, 
                                test_dataset,
                                target_domain: str,
                                batch_size: int = 32) -> Dict[str, Union[float, np.ndarray]]:
    """
    Evaluate ensemble performance on a test dataset.
    
    Args:
        ensemble: Ensemble model to evaluate
        test_dataset: Test dataset
        target_domain: Name of target domain being tested
        batch_size: Batch size for evaluation
        
    Returns:
        Dictionary containing evaluation metrics
    """
    predictions, labels = ensemble.predict_dataset(test_dataset, batch_size)
    
    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    
    # Generate classification report
    class_report = classification_report(labels, predictions, output_dict=True)
    
    # Generate confusion matrix
    conf_matrix = confusion_matrix(labels, predictions)
    
    results = {
        'target_domain': target_domain,
        'accuracy': accuracy,
        'predictions': predictions,
        'true_labels': labels,
        'classification_report': class_report,
        'confusion_matrix': conf_matrix,
        'ensemble_models': list(ensemble.models.keys()),
        'num_test_samples': len(predictions)
    }
    
    return results


def run_leave_one_out_experiments(dataset_path: str,
                                 models_path: str, 
                                 model_type: str = 'ResNet50DANN',
                                 batch_size: int = 32,
                                 use_binary_labels: bool = True) -> Dict[str, Dict]:
    """
    Run complete leave-one-out ensemble experiments.
    
    Args:
        dataset_path: Path to the dataset NPZ file
        models_path: Path to the models directory
        model_type: Type of model architecture
        batch_size: Batch size for evaluation
        use_binary_labels: Whether to use binary labels or multi-class
        
    Returns:
        Dictionary containing all experiment results
    """
    print("="*60)
    print("LEAVE-ONE-OUT ENSEMBLE EXPERIMENTS")
    print("="*60)
    
    # Load dataset
    print("\n1. Loading dataset...")
    images, labels, domain_text_list, sub_domain_text_list = load_cancer_dataset(
        dataset_path, use_binary_labels=use_binary_labels
    )
    
    # Load all models
    print("\n2. Loading domain-specific models...")
    models = load_all_domain_models(models_path, model_type)
    
    if len(models) == 0:
        raise ValueError("No models were successfully loaded!")
    
    # Create ensembles
    print("\n3. Creating leave-one-out ensembles...")
    ensembles = create_leave_one_out_ensembles(models)
    
    # Get transforms
    transform = get_standard_transforms()
    
    # Run experiments
    print("\n4. Running experiments...")
    results = {}
    
    for target_domain, ensemble in ensembles.items():
        print(f"\nEvaluating ensemble performance on '{target_domain}' domain...")
        
        # Create test dataset for target domain (leave-one-out: test on target domain only)
        try:
            from cancer_utils import create_leave_one_out_test_dataset
            test_dataset, metadata = create_leave_one_out_test_dataset(
                images=images,
                labels=labels,
                domain_text_list=domain_text_list,
                sub_domain_text_list=sub_domain_text_list,
                target_domain=target_domain,
                transform=transform,
                branched_mode=(model_type == 'ResNet50DANN')
            )
            
            print(f"  Test dataset size: {len(test_dataset)} samples")
            
            # Evaluate ensemble
            result = evaluate_ensemble_performance(
                ensemble=ensemble,
                test_dataset=test_dataset,
                target_domain=target_domain,
                batch_size=batch_size
            )
            
            result['metadata'] = metadata
            results[target_domain] = result
            
            print(f"  ✓ Accuracy: {result['accuracy']:.4f}")
            print(f"  ✓ Using models: {result['ensemble_models']}")
            
        except Exception as e:
            print(f"  ✗ Failed to evaluate {target_domain}: {e}")
            continue
    
    print(f"\n5. Completed experiments for {len(results)} domains")
    return results


def visualize_ensemble_results(results: Dict[str, Dict], save_path: Optional[str] = None):
    """
    Visualize ensemble experiment results.
    
    Args:
        results: Dictionary of experiment results from run_leave_one_out_experiments
        save_path: Optional path to save the plots
    """
    # Extract data for visualization
    domains = list(results.keys())
    accuracies = [results[domain]['accuracy'] for domain in domains]
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Leave-One-Out Ensemble Experiment Results', fontsize=16, fontweight='bold')
    
    # 1. Accuracy bar plot
    ax1 = axes[0, 0]
    bars = ax1.bar(domains, accuracies, color='skyblue', alpha=0.7, edgecolor='navy')
    ax1.set_title('Ensemble Accuracy by Target Domain')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Target Domain (Held Out)')
    ax1.tick_params(axis='x', rotation=45)
    ax1.set_ylim(0, 1)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Ensemble composition matrix
    ax2 = axes[0, 1]
    all_model_domains = set()
    for result in results.values():
        all_model_domains.update(result['ensemble_models'])
    
    all_model_domains = sorted(list(all_model_domains))
    composition_matrix = np.zeros((len(domains), len(all_model_domains)))
    
    for i, target_domain in enumerate(domains):
        ensemble_models = results[target_domain]['ensemble_models']
        for j, model_domain in enumerate(all_model_domains):
            composition_matrix[i, j] = 1 if model_domain in ensemble_models else 0
    
    im = ax2.imshow(composition_matrix, cmap='RdYlBu', aspect='auto')
    ax2.set_title('Ensemble Composition Matrix')
    ax2.set_xlabel('Models Used in Ensemble')
    ax2.set_ylabel('Target Domain (Held Out)')
    ax2.set_xticks(range(len(all_model_domains)))
    ax2.set_xticklabels(all_model_domains, rotation=45, ha='right')
    ax2.set_yticks(range(len(domains)))
    ax2.set_yticklabels(domains)
    
    # Add text annotations
    for i in range(len(domains)):
        for j in range(len(all_model_domains)):
            text = '✓' if composition_matrix[i, j] == 1 else '✗'
            ax2.text(j, i, text, ha="center", va="center", 
                    color="white" if composition_matrix[i, j] == 1 else "black", 
                    fontsize=12, fontweight='bold')
    
    # 3. Sample size information
    ax3 = axes[1, 0]
    sample_sizes = [results[domain]['num_test_samples'] for domain in domains]
    bars = ax3.bar(domains, sample_sizes, color='lightcoral', alpha=0.7, edgecolor='darkred')
    ax3.set_title('Test Set Size by Domain')
    ax3.set_ylabel('Number of Samples')
    ax3.set_xlabel('Target Domain')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, size in zip(bars, sample_sizes):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(sample_sizes)*0.01,
                str(size), ha='center', va='bottom', fontweight='bold')
    
    # 4. Performance summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Calculate summary statistics
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    min_acc = np.min(accuracies)
    max_acc = np.max(accuracies)
    
    best_domain = domains[np.argmax(accuracies)]
    worst_domain = domains[np.argmin(accuracies)]
    
    summary_text = f"""
    Ensemble Performance Summary
    ──────────────────────────────
    
    Mean Accuracy: {mean_acc:.3f} ± {std_acc:.3f}
    Best Performance: {best_domain} ({max_acc:.3f})
    Worst Performance: {worst_domain} ({min_acc:.3f})
    
    Total Domains: {len(domains)}
    Total Test Samples: {sum(sample_sizes):,}
    
    Ensemble Strategy: Average Predictions
    """
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved to: {save_path}")
    
    plt.show()
    
    # Print detailed results
    print("\n" + "="*80)
    print("DETAILED ENSEMBLE RESULTS")
    print("="*80)
    
    for target_domain in domains:
        result = results[target_domain]
        print(f"\nTarget Domain: {target_domain}")
        print(f"Ensemble Models: {', '.join(result['ensemble_models'])}")
        print(f"Test Samples: {result['num_test_samples']}")
        print(f"Accuracy: {result['accuracy']:.4f}")
        
        # Print per-class metrics
        class_report = result['classification_report']
        if '0' in class_report and '1' in class_report:
            print(f"Class 0 (Benign) F1: {class_report['0']['f1-score']:.4f}")
            print(f"Class 1 (Malignant) F1: {class_report['1']['f1-score']:.4f}")


def save_ensemble_results(results: Dict[str, Dict], save_dir: str):
    """
    Save ensemble experiment results to disk.
    
    Args:
        results: Dictionary of experiment results
        save_dir: Directory to save results
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save summary as JSON
    summary = {}
    for target_domain, result in results.items():
        summary[target_domain] = {
            'accuracy': float(result['accuracy']),
            'ensemble_models': result['ensemble_models'],
            'num_test_samples': int(result['num_test_samples']),
            'classification_report': result['classification_report']
        }
    
    with open(os.path.join(save_dir, 'ensemble_results_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save detailed results
    for target_domain, result in results.items():
        domain_dir = os.path.join(save_dir, f"target_{target_domain.replace(' ', '_')}")
        os.makedirs(domain_dir, exist_ok=True)
        
        np.save(os.path.join(domain_dir, 'predictions.npy'), result['predictions'])
        np.save(os.path.join(domain_dir, 'true_labels.npy'), result['true_labels'])
        np.save(os.path.join(domain_dir, 'confusion_matrix.npy'), result['confusion_matrix'])
    
    print(f"\nResults saved to: {save_dir}")


def compare_ensemble_vs_individual(results: Dict[str, Dict], 
                                 individual_results_path: Optional[str] = None):
    """
    Compare ensemble performance against individual model performance.
    
    Args:
        results: Ensemble experiment results
        individual_results_path: Optional path to individual model results for comparison
    """
    print("\n" + "="*80)
    print("ENSEMBLE vs INDIVIDUAL MODEL COMPARISON")
    print("="*80)
    
    for target_domain, result in results.items():
        ensemble_acc = result['accuracy']
        ensemble_models = result['ensemble_models']
        
        print(f"\nTarget Domain: {target_domain}")
        print(f"Ensemble Accuracy: {ensemble_acc:.4f}")
        print(f"Ensemble Models Used: {', '.join(ensemble_models)}")
        
        # If individual results are available, show comparison
        if individual_results_path and os.path.exists(individual_results_path):
            # This would require loading individual model results
            # Implementation depends on the format of individual results
            pass
        else:
            print("Individual model results not available for comparison")
        
        print("-" * 40)