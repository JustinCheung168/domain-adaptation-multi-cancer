"""
Cancer Dataset Utilities Module

This module contains utility functions for loading, preprocessing, and splitting 
cancer dataset for domain-specific experiments. Extracted from AC_Pipeline_clean.ipynb.

Key Features:
- Support for both single-branch (standard ResNet) and dual-branch (adversarial) architectures
- Domain-specific dataset splitting for leave-one-domain-out experiments
- Comprehensive evaluation and visualization utilities

Usage:
- For standard ResNet models: Use branched_mode=False or create_datasets_for_single_branch()
- For dual-branch adversarial models: Use branched_mode=True or create_datasets_for_dual_branch()
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from collections import Counter
from typing import List, Optional, Tuple, Dict
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report


class CustomImageDataset(Dataset):
    """Custom dataset class for image data with multiple labels."""
    
    def __init__(self, images, sub_domain_labels, cancer_binary_labels=None, transform=None, branched_mode=False):
        self.images = images
        self.sub_domain_labels = sub_domain_labels
        self.cancer_binary_labels = cancer_binary_labels if cancer_binary_labels is not None else np.zeros_like(sub_domain_labels)
        self.transform = transform
        self.branched_mode = branched_mode

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        
        # Apply transforms if provided
        if self.transform:
            # Transforms expect numpy array or PIL Image
            # ToTensor will handle conversion from numpy uint8 to float32 tensor
            if isinstance(image, np.ndarray):
                # Ensure image is in HWC format for transforms
                if len(image.shape) == 2:
                    # Grayscale, convert to RGB by stacking
                    image = np.stack([image, image, image], axis=-1)
            image = self.transform(image)
        else:
            # No transform, convert manually
            if isinstance(image, np.ndarray):
                if len(image.shape) == 3 and image.shape[-1] == 3:
                    # RGB image: HWC -> CHW
                    image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
                else:
                    # Grayscale, convert to RGB
                    image = torch.from_numpy(image).unsqueeze(0).repeat(3, 1, 1).float() / 255.0
        
        # Return different format based on mode
        if self.branched_mode:
            # For dual-branch architecture (matches TwoLabelDataset structure)
            return {
                "pixel_values": image,
                "labels1": int(self.sub_domain_labels[idx]),
                "labels2": int(self.cancer_binary_labels[idx])
            }
        else:
            # For standard single-branch architecture
            return {
                "pixel_values": image,
                "labels": self.sub_domain_labels[idx]
            }


def load_cancer_dataset(npz_path: str, use_binary_labels: bool = False) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Load the cancer dataset from .npz file.
    
    Args:
        npz_path: Path to the .npz dataset file
        use_binary_labels: If True, use binary cancer labels (cancer_binary), else multi-class labels (sub_domain)
        
    Returns:
        Tuple of (images, labels, domain_text_list, sub_domain_text_list)
    """
    print(f"Loading dataset from: {npz_path}")
    data = np.load(npz_path)
    
    # Extract arrays
    images = data['images']
    
    # Choose labels based on use_binary_labels flag
    if use_binary_labels:
        labels = data['cancer_binary'].astype(np.int64)  # Binary cancer labels (0, 1)
        print("Using binary cancer labels (0=negative, 1=positive)")
    else:
        labels = data['sub_domain'].astype(np.int64)  # Multi-class labels
        print("Using sub-domain classification labels")
    
    # Text metadata aligned 1:1 with dataset
    domain_text_list = data['domain_text'].astype(str).tolist()
    sub_domain_text_list = data['sub_domain_text'].astype(str).tolist()
    
    print(f"Loaded shapes: images={images.shape}, labels={labels.shape}")
    print(f"Label range: {min(labels)} to {max(labels)}")
    print(f"Label distribution: {dict(Counter(labels))}")
    print(f"Available domains: {sorted(set(domain_text_list))}")
    print(f"Available sub-domains: {len(set(sub_domain_text_list))} unique")
    
    return images, labels, domain_text_list, sub_domain_text_list


def get_domain_splits(domain_text_list: List[str], 
                     target_domain: str,
                     train_val_ratio: float = 0.8,
                     include_source_test: bool = False,
                     source_test_ratio: float = 0.1,
                     seed: int = 42) -> Tuple[List[int], List[int], List[int], List[int]]:
    """
    Create domain-specific train/val/test splits where:
    - Train/Val: Samples from target_domain
    - Test: All samples from other domains + optional source domain test set
    
    Args:
        domain_text_list: List of domain labels for each sample
        target_domain: Domain to use for training/validation
        train_val_ratio: Ratio of target domain samples to use for training (rest for validation)
        include_source_test: If True, create a test set from source domain
        source_test_ratio: Ratio of source domain to reserve for testing
        seed: Random seed for reproducible splits
        
    Returns:
        Tuple of (train_indices, val_indices, test_indices, source_test_indices)
    """
    # Set seed for reproducible splits
    rng = np.random.RandomState(seed)
    
    domain_array = np.array(domain_text_list)
    
    # Get indices for target domain and other domains
    target_indices = np.where(domain_array == target_domain)[0]
    other_indices = np.where(domain_array != target_domain)[0]
    
    # Create source domain test set if requested
    source_test_indices = []
    if include_source_test:
        # Reserve some target domain samples for testing
        rng.shuffle(target_indices)
        n_source_test = int(len(target_indices) * source_test_ratio)
        source_test_indices = target_indices[:n_source_test].tolist()
        target_indices = target_indices[n_source_test:]  # Remaining for train/val
    
    # Split remaining target domain into train/val
    rng.shuffle(target_indices)
    n_train = int(len(target_indices) * train_val_ratio)
    
    train_indices = target_indices[:n_train]
    val_indices = target_indices[n_train:]
    test_indices = other_indices
    
    print(f"Domain '{target_domain}' splits:")
    print(f"  Train: {len(train_indices)} samples (from {target_domain})")
    print(f"  Val:   {len(val_indices)} samples (from {target_domain})")
    print(f"  Test:  {len(test_indices)} samples (from other domains)")
    
    if include_source_test and source_test_indices:
        print(f"  Source Test: {len(source_test_indices)} samples (from {target_domain})")
    
    # Print test domain distribution
    test_domains = Counter(domain_array[test_indices])
    print(f"  Test domain distribution: {dict(test_domains)}")
    
    return train_indices.tolist(), val_indices.tolist(), test_indices.tolist(), source_test_indices


def create_domain_datasets(images: np.ndarray,
                          labels: np.ndarray,
                          domain_text_list: List[str],
                          sub_domain_text_list: List[str],
                          target_domain: str,
                          train_val_ratio: float = 0.8,
                          transform: Optional[transforms.Compose] = None,
                          branched_mode: bool = False,
                          include_source_test: bool = False,
                          source_test_ratio: float = 0.1,
                          seed: int = 42) -> Tuple[CustomImageDataset, CustomImageDataset, CustomImageDataset, Optional[CustomImageDataset], Dict]:
    """
    Create train/val/test datasets for a specific target domain.
    
    Args:
        images: Image data array
        labels: Label array
        domain_text_list: List of domain labels
        sub_domain_text_list: List of sub-domain labels
        target_domain: Domain to use for training/validation
        train_val_ratio: Ratio for train/val split within target domain
        transform: Optional transforms to apply
        branched_mode: If True, return datasets compatible with dual-branch architecture
        include_source_test: If True, create a test set from source domain
        source_test_ratio: Ratio of source domain to reserve for testing
        seed: Random seed for reproducible splits
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset, source_test_dataset, metadata_dict)
    """
    # Get domain-specific splits with optional source test set
    train_idx, val_idx, test_idx, source_test_idx = get_domain_splits(
        domain_text_list, target_domain, train_val_ratio, include_source_test, source_test_ratio, seed
    )
    
    # Create datasets
    train_dataset = CustomImageDataset(
        images[train_idx], 
        labels[train_idx],
        transform=transform,
        branched_mode=branched_mode
    )
    
    val_dataset = CustomImageDataset(
        images[val_idx], 
        labels[val_idx],
        transform=transform,
        branched_mode=branched_mode
    )
    
    test_dataset = CustomImageDataset(
        images[test_idx], 
        labels[test_idx],
        transform=transform,
        branched_mode=branched_mode
    )
    
    # Create source test dataset if requested
    source_test_dataset = None
    if include_source_test and source_test_idx:
        source_test_dataset = CustomImageDataset(
            images[source_test_idx], 
            labels[source_test_idx],
            transform=transform,
            branched_mode=branched_mode
        )
    
    # Create metadata
    metadata = {
        'target_domain': target_domain,
        'train_domains': [domain_text_list[i] for i in train_idx],
        'val_domains': [domain_text_list[i] for i in val_idx],
        'test_domains': [domain_text_list[i] for i in test_idx],
        'train_subdomains': [sub_domain_text_list[i] for i in train_idx],
        'val_subdomains': [sub_domain_text_list[i] for i in val_idx],
        'test_subdomains': [sub_domain_text_list[i] for i in test_idx],
        'train_size': len(train_idx),
        'val_size': len(val_idx),
        'test_size': len(test_idx),
        'source_test_size': len(source_test_idx) if source_test_idx else 0,
        'include_source_test': include_source_test
    }
    
    if include_source_test and source_test_idx:
        metadata['source_test_domains'] = [domain_text_list[i] for i in source_test_idx]
        metadata['source_test_subdomains'] = [sub_domain_text_list[i] for i in source_test_idx]
    
    return train_dataset, val_dataset, test_dataset, source_test_dataset, metadata


def get_standard_transforms() -> transforms.Compose:
    """Get the standard transform pipeline for cancer images."""
    return transforms.Compose([
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def create_datasets_for_single_branch(images: np.ndarray,
                                    labels: np.ndarray,
                                    domain_text_list: List[str],
                                    sub_domain_text_list: List[str],
                                    target_domain: str,
                                    train_val_ratio: float = 0.8,
                                    transform: Optional[transforms.Compose] = None) -> Tuple[CustomImageDataset, CustomImageDataset, CustomImageDataset, Dict]:
    """
    Convenience function to create datasets for single-branch (standard ResNet) models.
    This is equivalent to calling create_domain_datasets with branched_mode=False.
    """
    return create_domain_datasets(
        images=images,
        labels=labels,
        domain_text_list=domain_text_list,
        sub_domain_text_list=sub_domain_text_list,
        target_domain=target_domain,
        train_val_ratio=train_val_ratio,
        transform=transform,
        branched_mode=False
    )


def create_datasets_for_dual_branch(images: np.ndarray,
                                  labels: np.ndarray,
                                  domain_text_list: List[str],
                                  sub_domain_text_list: List[str],
                                  target_domain: str,
                                  train_val_ratio: float = 0.8,
                                  transform: Optional[transforms.Compose] = None) -> Tuple[CustomImageDataset, CustomImageDataset, CustomImageDataset, Dict]:
    """
    Convenience function to create datasets for dual-branch (adversarial ResNet) models.
    This is equivalent to calling create_domain_datasets with branched_mode=True.
    """
    return create_domain_datasets(
        images=images,
        labels=labels,
        domain_text_list=domain_text_list,
        sub_domain_text_list=sub_domain_text_list,
        target_domain=target_domain,
        train_val_ratio=train_val_ratio,
        transform=transform,
        branched_mode=True
    )


def print_dataset_summary(metadata: Dict):
    """Print a summary of the dataset splits."""
    print(f"\nDataset Summary for '{metadata['target_domain']}':")
    print(f"  Training:   {metadata['train_size']} samples")
    print(f"  Validation: {metadata['val_size']} samples")
    print(f"  Testing:    {metadata['test_size']} samples")
    
    if metadata.get('include_source_test', False) and metadata.get('source_test_size', 0) > 0:
        print(f"  Source Test: {metadata['source_test_size']} samples")
    
    # Print label distributions
    print(f"\nDomain distributions:")
    print(f"  Train domains: {Counter(metadata['train_domains'])}")
    print(f"  Val domains:   {Counter(metadata['val_domains'])}")
    print(f"  Test domains:  {Counter(metadata['test_domains'])}")
    
    if metadata.get('include_source_test', False) and 'source_test_domains' in metadata:
        print(f"  Source Test domains: {Counter(metadata['source_test_domains'])}")


def evaluate_model_on_domains(model, test_dataset, test_metadata, device='cuda'):
    """
    Evaluate a model on test set and provide domain-wise breakdown.
    
    Args:
        model: Trained model
        test_dataset: Test dataset
        test_metadata: Test metadata with domain information
        device: Device to run evaluation on
        
    Returns:
        Dictionary with evaluation results
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    # Get predictions
    with torch.no_grad():
        for i in range(len(test_dataset)):
            sample = test_dataset[i]
            pixel_values = sample['pixel_values'].unsqueeze(0).to(device)
            labels = sample['labels']
            
            outputs = model(pixel_values)
            pred = torch.argmax(outputs.logits, dim=-1).cpu().numpy()[0]
            
            all_preds.append(pred)
            all_labels.append(labels)
    
    # Overall accuracy
    overall_acc = np.mean(np.array(all_preds) == np.array(all_labels))
    
    # Domain-wise accuracy
    domain_results = {}
    test_domains = test_metadata['test_domains']
    
    for domain in set(test_domains):
        domain_indices = [i for i, d in enumerate(test_domains) if d == domain]
        if domain_indices:
            domain_preds = [all_preds[i] for i in domain_indices]
            domain_labels = [all_labels[i] for i in domain_indices]
            domain_acc = np.mean(np.array(domain_preds) == np.array(domain_labels))
            domain_results[domain] = {
                'accuracy': domain_acc,
                'count': len(domain_indices),
                'predictions': domain_preds,
                'labels': domain_labels
            }
    
    return {
        'overall_accuracy': overall_acc,
        'domain_results': domain_results,
        'all_predictions': all_preds,
        'all_labels': all_labels
    }


def evaluate_comprehensive(model, test_dataset, source_test_dataset, metadata, device='cuda'):
    """
    Comprehensive evaluation including both cross-domain and source domain test sets.
    
    Args:
        model: Trained model
        test_dataset: Cross-domain test dataset
        source_test_dataset: Source domain test dataset (can be None)
        metadata: Dataset metadata
        device: Device to run evaluation on
        
    Returns:
        Dictionary with comprehensive evaluation results
    """
    results = {}
    
    # Evaluate on cross-domain test set
    if test_dataset:
        cross_domain_results = evaluate_model_on_domains(model, test_dataset, metadata, device)
        results['cross_domain'] = cross_domain_results
    
    # Evaluate on source domain test set
    if source_test_dataset and metadata.get('include_source_test', False):
        model.eval()
        source_preds = []
        source_labels = []
        
        with torch.no_grad():
            for i in range(len(source_test_dataset)):
                sample = source_test_dataset[i]
                pixel_values = sample['pixel_values'].unsqueeze(0).to(device)
                label = sample['labels']
                
                outputs = model(pixel_values)
                pred = torch.argmax(outputs.logits, dim=-1).cpu().numpy()[0]
                
                source_preds.append(pred)
                source_labels.append(label)
        
        source_acc = np.mean(np.array(source_preds) == np.array(source_labels))
        results['source_domain'] = {
            'accuracy': source_acc,
            'count': len(source_preds),
            'predictions': source_preds,
            'labels': source_labels,
            'domain': metadata['target_domain']
        }
    
    return results


def plot_domain_results(results_dict: Dict[str, Dict], save_path: Optional[str] = None):
    """
    Plot domain-wise results across different training domains.
    Now includes source domain test results for a complete heatmap!
    
    Args:
        results_dict: Dictionary mapping training_domain -> evaluation results
        save_path: Optional path to save the plot
    """
    # Extract data for plotting
    training_domains = list(results_dict.keys())
    test_domains = set()
    
    # Collect all unique test domains (including source domains)
    for result in results_dict.values():
        test_domains.update(result['domain_results'].keys())
    
    test_domains = sorted(list(test_domains))
    
    print(f"Creating COMPLETE heatmap:")
    print(f"  Training domains: {training_domains}")
    print(f"  Test domains: {test_domains}")
    
    # Create accuracy matrix
    accuracy_matrix = np.zeros((len(training_domains), len(test_domains)))
    
    for i, train_domain in enumerate(training_domains):
        result = results_dict[train_domain]
        for j, test_domain in enumerate(test_domains):
            if test_domain in result['domain_results']:
                accuracy_matrix[i, j] = result['domain_results'][test_domain]['accuracy']
            else:
                accuracy_matrix[i, j] = np.nan
    
    # Create COMPLETE heatmap with better color scheme
    fig, ax = plt.subplots(figsize=(14, 10))
    im = ax.imshow(accuracy_matrix, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(test_domains)))
    ax.set_yticks(np.arange(len(training_domains)))
    ax.set_xticklabels(test_domains, rotation=45, ha='right')
    ax.set_yticklabels(training_domains)
    
    # Add text annotations with black text for all values
    for i in range(len(training_domains)):
        for j in range(len(test_domains)):
            if not np.isnan(accuracy_matrix[i, j]):
                # Use different text weight for source domain (diagonal) vs cross-domain
                is_source_domain = training_domains[i] == test_domains[j]
                weight = "bold" if is_source_domain else "normal"
                
                text = ax.text(j, i, f'{accuracy_matrix[i, j]:.3f}',
                             ha="center", va="center", color="black", 
                             fontsize=10, weight=weight)
    
    ax.set_title('COMPLETE Domain Generalization Results\n(Training Domain vs Test Domain Accuracy)\n(Bold = Source Domain, Regular = Cross-Domain)', 
                 fontsize=14)
    ax.set_xlabel('Test Domain', fontsize=12)
    ax.set_ylabel('Training Domain', fontsize=12)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Accuracy', rotation=270, labelpad=15)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()
    
    # Print COMPLETE summary table
    print("\nCOMPLETE Domain Generalization Summary:")
    print("=" * 100)
    print(f"{'Training Domain':<20} {'Source Acc':<12} {'Cross-Dom Acc':<15} {'Best Test':<20} {'Worst Test':<20}")
    print("=" * 100)
    
    for train_domain in training_domains:
        result = results_dict[train_domain]
        
        domain_accs = {k: v['accuracy'] for k, v in result['domain_results'].items()}
        
        # Separate source domain accuracy from cross-domain accuracies
        source_acc = domain_accs.get(train_domain, None)
        cross_domain_accs = {k: v for k, v in domain_accs.items() if k != train_domain}
        
        # Calculate cross-domain average
        cross_dom_avg = np.mean(list(cross_domain_accs.values())) if cross_domain_accs else None
        
        if domain_accs:
            best_domain = max(domain_accs.items(), key=lambda x: x[1])
            worst_domain = min(domain_accs.items(), key=lambda x: x[1])
            
            source_str = f"{source_acc:.3f}" if source_acc is not None else "N/A"
            cross_str = f"{cross_dom_avg:.3f}" if cross_dom_avg is not None else "N/A"
            
            print(f"{train_domain:<20} {source_str:<12} {cross_str:<15} {best_domain[0]:<20} {worst_domain[0]:<20}")
        else:
            print(f"{train_domain:<20} {'N/A':<12} {'N/A':<15} {'N/A':<20} {'N/A':<20}")
    
    print("\nLegend:")
    print("  Source Acc: Accuracy on same domain test set")
    print("  Cross-Dom Acc: Average accuracy across other domains")
    print("  Best/Worst Test: Best/worst performing test domain overall")


def save_experiment_results(results_dict: Dict[str, Dict], save_dir: str):
    """
    Save experiment results to disk.
    
    Args:
        results_dict: Dictionary of experiment results
        save_dir: Directory to save results
    """
    import os
    import json
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Save summary results as JSON
    summary = {}
    for train_domain, result in results_dict.items():
        summary[train_domain] = {
            'overall_accuracy': float(result['overall_accuracy']),
            'domain_accuracies': {k: float(v['accuracy']) for k, v in result['domain_results'].items()}
        }
    
    with open(os.path.join(save_dir, 'experiment_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save detailed results as numpy arrays
    for train_domain, result in results_dict.items():
        domain_dir = os.path.join(save_dir, f"training_{train_domain.replace(' ', '_')}")
        os.makedirs(domain_dir, exist_ok=True)
        
        np.save(os.path.join(domain_dir, 'predictions.npy'), result['all_predictions'])
        np.save(os.path.join(domain_dir, 'labels.npy'), result['all_labels'])
    
    print(f"Results saved to: {save_dir}")


def create_leave_one_out_test_dataset(images: np.ndarray,
                                     labels: np.ndarray,
                                     domain_text_list: List[str],
                                     sub_domain_text_list: List[str],
                                     target_domain: str,
                                     transform: Optional[transforms.Compose] = None,
                                     branched_mode: bool = False) -> Tuple[CustomImageDataset, Dict]:
    """
    Create a test dataset containing ONLY samples from the target domain
    for leave-one-out ensemble evaluation.
    
    Args:
        images: Image data array
        labels: Label array  
        domain_text_list: List of domain labels
        sub_domain_text_list: List of sub-domain labels
        target_domain: Domain to create test dataset for (held-out domain)
        transform: Optional transforms to apply
        branched_mode: If True, return dataset compatible with dual-branch architecture
        
    Returns:
        Tuple of (test_dataset, metadata_dict)
    """
    domain_array = np.array(domain_text_list)
    
    # Get indices for target domain ONLY
    target_indices = np.where(domain_array == target_domain)[0]
    
    # Create test dataset from target domain samples only
    test_dataset = CustomImageDataset(
        images[target_indices], 
        labels[target_indices],
        transform=transform,
        branched_mode=branched_mode
    )
    
    # Create metadata
    from collections import Counter
    target_labels = labels[target_indices]
    label_counts = Counter(target_labels)
    
    metadata = {
        'target_domain': target_domain,
        'test_samples': len(target_indices),
        'test_label_distribution': dict(label_counts),
        'test_domains': {target_domain: len(target_indices)}
    }
    
    print(f"Leave-one-out test dataset for '{target_domain}':")
    print(f"  Test samples: {len(target_indices)} (from {target_domain} domain only)")
    print(f"  Label distribution: {dict(label_counts)}")
    
    return test_dataset, metadata


def load_adenocarcinoma_dataset(npz_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Load the adenocarcinoma dataset from .npz file.
    
    The adenocarcinoma dataset has the following structure:
    - images: (N, H, W, 3) RGB images
    - sub_domain: class IDs for tissue types (labels1 in original)
    - cancer_binary: binary labels 0=benign, 1=malignant (labels2 in original)
    - domain_text: domain names (e.g., "Breast Cancer", "Colon Cancer")
    - sub_domain_text: tissue type names (e.g., "breast_benign", "colon_aca")
    
    Args:
        npz_path: Path to the adenocarcinoma_dataset.npz file
        
    Returns:
        Tuple of (images, sub_domain_labels, cancer_binary_labels, domain_text_list, sub_domain_text_list)
    """
    print(f"Loading adenocarcinoma dataset from: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    
    # Extract arrays with correct keys for adenocarcinoma dataset
    images = data['images']
    sub_domain_labels = data['sub_domain'].astype(np.int64)  # Class IDs
    cancer_binary_labels = data['cancer_binary'].astype(np.int64)  # Binary 0/1
    
    # Text metadata
    domain_text_list = data['domain_text'].astype(str).tolist()
    sub_domain_text_list = data['sub_domain_text'].astype(str).tolist()
    
    print(f"Loaded shapes: images={images.shape}, sub_domain={sub_domain_labels.shape}, cancer_binary={cancer_binary_labels.shape}")
    print(f"Sub-domain label range: {min(sub_domain_labels)} to {max(sub_domain_labels)}")
    print(f"Cancer binary label distribution: {dict(Counter(cancer_binary_labels))}")
    print(f"Available domains: {sorted(set(domain_text_list))}")
    print(f"Available sub-domains: {sorted(set(sub_domain_text_list))}")
    
    return images, sub_domain_labels, cancer_binary_labels, domain_text_list, sub_domain_text_list


def preprocess_for_branched_resnet(images: np.ndarray,
                                   cancer_binary_labels: np.ndarray,
                                   domain_text_list: List[str],
                                   target_domain: str,
                                   normalize: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Preprocess adenocarcinoma dataset for branched ResNet model.
    Memory-efficient: Skips normalization to avoid creating float32 copy of entire dataset.
    
    Converts the dataset to have:
    - labels1: cancer_binary (0=benign, 1=malignant) - primary classification task
    - labels2: binary target domain indicator (1 if domain==target_domain, 0 otherwise)
    
    Args:
        images: Image data array (N, H, W, 3)
        cancer_binary_labels: Binary cancer labels (0=benign, 1=malignant)
        domain_text_list: List of domain names for each sample
        target_domain: Target domain for labels2 encoding
        normalize: If True, normalization will happen on-the-fly during training
        
    Returns:
        Tuple of (preprocessed_images, labels1, labels2)
    """
    # Memory-efficient: Keep images as uint8, normalize on-the-fly with transforms
    # This avoids creating a 40,000 x 224 x 224 x 3 x 4 bytes = ~24GB float32 array
    print(f"Keeping images as {images.dtype} to reduce RAM usage...")
    print(f"  Images will be normalized on-the-fly during training")
    images_processed = images
    
    # labels1 = cancer_binary (primary task) - avoid copy if already int64
    labels1 = cancer_binary_labels if cancer_binary_labels.dtype == np.int64 else cancer_binary_labels.astype(np.int64)
    
    # labels2 = binary indicator for target_domain
    domain_array = np.array(domain_text_list)
    labels2 = (domain_array == target_domain).astype(np.int64)
    
    print(f"Preprocessed for branched ResNet:")
    print(f"  Images shape: {images_processed.shape}, dtype: {images_processed.dtype}")
    print(f"  labels1 (cancer_binary) distribution: {dict(Counter(labels1))}")
    print(f"  labels2 (target_domain={target_domain}) distribution: {dict(Counter(labels2))}")
    
    return images_processed, labels1, labels2


def get_branched_resnet_transforms() -> transforms.Compose:
    """
    Get the standard transform pipeline for branched ResNet model.
    
    This transform converts uint8 images [0-255] to normalized float32 tensors [-1, 1].
    Normalization happens on-the-fly during training to reduce memory usage.
    
    Returns:
        transforms.Compose with ToTensor (scales to [0,1]) and Normalize (to [-1,1])
    """
    # ToTensor automatically converts uint8 [0-255] to float32 [0.0-1.0]
    # Then Normalize converts [0.0-1.0] to [-1.0, 1.0] using: (x - 0.5) / 0.5
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def create_branched_dataset(images: np.ndarray,
                           labels1: np.ndarray,
                           labels2: np.ndarray,
                           transform: Optional[transforms.Compose] = None) -> CustomImageDataset:
    """
    Create a CustomImageDataset for branched ResNet model.
    
    Args:
        images: Preprocessed image data
        labels1: Primary labels (cancer_binary)
        labels2: Secondary labels (target_domain indicator)
        transform: Optional transforms to apply
        
    Returns:
        CustomImageDataset configured for branched mode
    """
    dataset = CustomImageDataset(
        images=images,
        sub_domain_labels=labels1,
        cancer_binary_labels=labels2,
        transform=transform,
        branched_mode=True
    )
    
    print(f"Created branched dataset:")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Sample structure: {list(dataset[0].keys())}")
    
    return dataset


def split_dataset_stratified(dataset: CustomImageDataset,
                            domain_text_list: List[str],
                            sub_domain_text_list: List[str],
                            train_size: float = 0.7,
                            val_size: float = 0.15,
                            test_size: float = 0.15,
                            stratify_on: str = "labels1",
                            seed: int = 42) -> Tuple[CustomImageDataset, CustomImageDataset, CustomImageDataset, 
                                                     List[str], List[str], List[str], 
                                                     List[str], List[str], List[str]]:
    """
    Split dataset into train/val/test with stratification while preserving text metadata.
    Memory-efficient: Uses array slicing instead of copying.
    
    Args:
        dataset: CustomImageDataset to split
        domain_text_list: List of domain labels for each sample
        sub_domain_text_list: List of sub-domain labels for each sample
        train_size: Fraction for training set
        val_size: Fraction for validation set
        test_size: Fraction for test set
        stratify_on: Attribute to stratify on ("labels1" or "labels2")
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_ds, val_ds, test_ds, 
                 train_domains, val_domains, test_domains,
                 train_subdomains, val_subdomains, test_subdomains)
    """
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError("train_size + val_size + test_size must sum to 1.0")
    
    print(f"  Performing memory-efficient stratified split...")
    
    rng = np.random.default_rng(seed)
    N = len(dataset)
    
    # Get stratification labels - use existing arrays, don't copy
    if stratify_on == "labels1":
        strat_labels = dataset.sub_domain_labels
    elif stratify_on == "labels2":
        strat_labels = dataset.cancer_binary_labels
    else:
        raise ValueError("stratify_on must be 'labels1' or 'labels2'")
    
    # Group indices by label
    label_to_indices = {}
    for idx, label in enumerate(strat_labels):
        label_to_indices.setdefault(int(label), []).append(idx)
    
    # Split each label group
    train_idx, val_idx, test_idx = [], [], []
    
    for label, indices in label_to_indices.items():
        indices = np.array(indices)
        rng.shuffle(indices)
        
        n = len(indices)
        n_train = int(np.floor(train_size * n))
        n_val = int(np.floor(val_size * n))
        
        train_idx.extend(indices[:n_train].tolist())
        val_idx.extend(indices[n_train:n_train + n_val].tolist())
        test_idx.extend(indices[n_train + n_val:].tolist())
    
    # Shuffle final splits
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    
    # Convert to arrays for metadata alignment
    domain_array = np.array(domain_text_list)
    subdomain_array = np.array(sub_domain_text_list)
    
    # Create subset datasets
    def build_subset(indices):
        return CustomImageDataset(
            images=dataset.images[indices],
            sub_domain_labels=dataset.sub_domain_labels[indices],
            cancer_binary_labels=dataset.cancer_binary_labels[indices],
            transform=dataset.transform,
            branched_mode=dataset.branched_mode
        )
    
    train_ds = build_subset(train_idx)
    val_ds = build_subset(val_idx)
    test_ds = build_subset(test_idx)
    
    # Extract aligned text metadata
    train_domains = domain_array[train_idx].tolist()
    val_domains = domain_array[val_idx].tolist()
    test_domains = domain_array[test_idx].tolist()
    
    train_subdomains = subdomain_array[train_idx].tolist()
    val_subdomains = subdomain_array[val_idx].tolist()
    test_subdomains = subdomain_array[test_idx].tolist()
    
    print(f"Stratified split (on {stratify_on}):")
    print(f"  Train: {len(train_ds)} samples")
    print(f"  Val:   {len(val_ds)} samples")
    print(f"  Test:  {len(test_ds)} samples")
    print(f"  Train domain distribution: {Counter(train_domains)}")
    print(f"  Val domain distribution: {Counter(val_domains)}")
    print(f"  Test domain distribution: {Counter(test_domains)}")
    
    return (train_ds, val_ds, test_ds,
            train_domains, val_domains, test_domains,
            train_subdomains, val_subdomains, test_subdomains)


def cancer_preprocess(npz_path: str,
                     target_domain: str,
                     train_size: float = 0.7,
                     val_size: float = 0.15,
                     test_size: float = 0.15,
                     stratify_on: str = "labels1",
                     normalize: bool = True,
                     seed: int = 42) -> Tuple[CustomImageDataset, CustomImageDataset, CustomImageDataset, Dict]:
    """
    Complete preprocessing pipeline for adenocarcinoma dataset to be used with branched ResNet.
    
    This function performs all necessary preprocessing steps:
    1. Loads the adenocarcinoma dataset
    2. Preprocesses images (normalization)
    3. Converts labels: labels1=cancer_binary, labels2=target_domain_indicator
    4. Creates CustomImageDataset with transforms
    5. Splits into train/val/test with stratification
    6. Returns datasets ready for training
    
    Args:
        npz_path: Path to adenocarcinoma_dataset.npz file
        target_domain: Target domain for labels2 encoding (e.g., "Breast Cancer")
        train_size: Fraction for training set (default: 0.7)
        val_size: Fraction for validation set (default: 0.15)
        test_size: Fraction for test set (default: 0.15)
        stratify_on: Stratify splits on "labels1" (cancer_binary) or "labels2" (target_domain)
        normalize: If True, normalize images to [-1, 1] range
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset, metadata_dict)
        
    Example:
        >>> train_ds, val_ds, test_ds, metadata = cancer_preprocess(
        ...     'data/multi_cancer/adenocarcinoma_dataset.npz',
        ...     target_domain='Breast Cancer',
        ...     train_size=0.7,
        ...     val_size=0.15,
        ...     test_size=0.15
        ... )
        >>> print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    """
    print("=" * 80)
    print("CANCER DATASET PREPROCESSING PIPELINE")
    print("=" * 80)
    
    # Step 1: Load adenocarcinoma dataset
    print("\n[Step 1/5] Loading adenocarcinoma dataset...")
    images, sub_domain_labels, cancer_binary_labels, domain_text_list, sub_domain_text_list = \
        load_adenocarcinoma_dataset(npz_path)
    
    # Step 2: Preprocess for branched ResNet
    print(f"\n[Step 2/5] Preprocessing for branched ResNet (target_domain='{target_domain}')...")
    images_processed, labels1, labels2 = preprocess_for_branched_resnet(
        images=images,
        cancer_binary_labels=cancer_binary_labels,
        domain_text_list=domain_text_list,
        target_domain=target_domain,
        normalize=normalize
    )
    
    # Step 3: Get transforms
    print("\n[Step 3/5] Creating image transforms...")
    transform = get_branched_resnet_transforms()
    print(f"  Transforms: {transform}")
    
    # Step 4: Create full dataset
    print("\n[Step 4/5] Creating CustomImageDataset...")
    full_dataset = create_branched_dataset(
        images=images_processed,
        labels1=labels1,
        labels2=labels2,
        transform=transform
    )
    
    # Step 5: Split dataset
    print(f"\n[Step 5/5] Splitting dataset (train={train_size}, val={val_size}, test={test_size})...")
    train_ds, val_ds, test_ds, train_domains, val_domains, test_domains, \
        train_subdomains, val_subdomains, test_subdomains = split_dataset_stratified(
        dataset=full_dataset,
        domain_text_list=domain_text_list,
        sub_domain_text_list=sub_domain_text_list,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        stratify_on=stratify_on,
        seed=seed
    )
    
    # Create metadata
    metadata = {
        'target_domain': target_domain,
        'train_size': len(train_ds),
        'val_size': len(val_ds),
        'test_size': len(test_ds),
        'train_domains': train_domains,
        'val_domains': val_domains,
        'test_domains': test_domains,
        'train_subdomains': train_subdomains,
        'val_subdomains': val_subdomains,
        'test_subdomains': test_subdomains,
        'num_classes_labels1': int(np.max(labels1)) + 1,
        'num_classes_labels2': int(np.max(labels2)) + 1,
        'labels1_distribution_train': dict(Counter(train_ds.sub_domain_labels)),
        'labels2_distribution_train': dict(Counter(train_ds.cancer_binary_labels)),
        'stratified_on': stratify_on,
        'seed': seed
    }
    
    print("\n" + "=" * 80)
    print("PREPROCESSING COMPLETE")
    print("=" * 80)
    print(f"\nDatasets ready for branched ResNet training:")
    print(f"  Train: {len(train_ds)} samples")
    print(f"  Val:   {len(val_ds)} samples")
    print(f"  Test:  {len(test_ds)} samples")
    print(f"\nLabels configuration:")
    print(f"  labels1: cancer_binary (0=benign, 1=malignant)")
    print(f"  labels2: target_domain indicator (1={target_domain}, 0=other)")
    print(f"  num_d1_classes: {metadata['num_classes_labels1']}")
    print(f"  num_d2_classes: {metadata['num_classes_labels2']}")
    
    return train_ds, val_ds, test_ds, metadata