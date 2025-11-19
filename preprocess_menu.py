#!/usr/bin/env python3
"""
Interactive Preprocessing Menu for Multi-Cancer Dataset

This script provides an interactive terminal menu to:
1. Select a target domain from the adenocarcinoma dataset
2. Preprocess the dataset with proper label configuration
3. Save train/val/test splits as .npz files compatible with the experiment config framework

The output .npz files contain:
- images: preprocessed image data
- labels1: cancer_binary (0=benign, 1=malignant) - primary classification task
- labels2: binary target domain indicator (1=target_domain, 0=other) - adversarial task
"""

import os
import sys
import json
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np

# Add src to path for imports
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from cancer_utils import cancer_preprocess


def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header():
    """Print the application header."""
    print("=" * 80)
    print(" " * 20 + "MULTI-CANCER DATASET PREPROCESSOR")
    print("=" * 80)
    print()


def get_available_domains(npz_path: str) -> List[str]:
    """
    Extract available domains from the dataset.
    
    Args:
        npz_path: Path to the adenocarcinoma_dataset.npz file
        
    Returns:
        Sorted list of unique domain names
    """
    try:
        data = np.load(npz_path, allow_pickle=True)
        domain_text_list = data['domain_text'].astype(str).tolist()
        domains = sorted(set(domain_text_list))
        return domains
    except FileNotFoundError:
        print(f"ERROR: Dataset file not found: {npz_path}")
        sys.exit(1)
    except KeyError as e:
        print(f"ERROR: Required key missing in dataset: {e}")
        sys.exit(1)


def display_menu(domains: List[str]) -> int:
    """
    Display domain selection menu and get user choice.
    
    Args:
        domains: List of available domain names
        
    Returns:
        Selected domain index (0-based)
    """
    print("\nAVAILABLE CANCER DOMAINS:")
    print("-" * 80)
    
    for idx, domain in enumerate(domains, 1):
        print(f"  [{idx}] {domain}")
    
    print(f"  [0] Exit")
    print("-" * 80)
    
    while True:
        try:
            choice = input("\nSelect target domain number: ").strip()
            choice_num = int(choice)
            
            if choice_num == 0:
                print("\nExiting preprocessor. Goodbye!")
                sys.exit(0)
            
            if 1 <= choice_num <= len(domains):
                return choice_num - 1  # Convert to 0-based index
            else:
                print(f"Invalid choice. Please enter a number between 0 and {len(domains)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Exiting...")
            sys.exit(0)


def get_split_ratios() -> Tuple[float, float, float]:
    """
    Get train/val/test split ratios from user.
    
    Returns:
        Tuple of (train_size, val_size, test_size)
    """
    print("\n" + "=" * 80)
    print("DATASET SPLIT CONFIGURATION")
    print("=" * 80)
    print("\nDefault split ratios: Train=70%, Val=15%, Test=15%")
    
    use_default = input("Use default split ratios? (Y/n): ").strip().lower()
    
    if use_default in ['', 'y', 'yes']:
        return 0.7, 0.15, 0.15
    
    print("\nEnter custom split ratios (must sum to 1.0):")
    
    while True:
        try:
            train_size = float(input("  Train size (0.0-1.0): ").strip())
            val_size = float(input("  Val size (0.0-1.0): ").strip())
            test_size = float(input("  Test size (0.0-1.0): ").strip())
            
            if not (0.0 < train_size < 1.0 and 0.0 < val_size < 1.0 and 0.0 < test_size < 1.0):
                print("ERROR: All sizes must be between 0.0 and 1.0")
                continue
            
            if not np.isclose(train_size + val_size + test_size, 1.0):
                print(f"ERROR: Sizes must sum to 1.0 (current sum: {train_size + val_size + test_size:.3f})")
                continue
            
            return train_size, val_size, test_size
        except ValueError:
            print("ERROR: Invalid input. Please enter decimal numbers.")
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Exiting...")
            sys.exit(0)


def find_split_index_files(output_dir: Path) -> List[Path]:
    """
    Find existing split index files in the output directory.
    
    Returns:
        List of paths to .json split index files
    """
    if not output_dir.exists():
        return []
    return sorted(output_dir.glob("*_split_indices.json"))


def load_split_indices(indices_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load split indices from JSON file.
    
    Returns:
        Tuple of (train_indices, val_indices, test_indices)
    """
    with open(indices_path, 'r') as f:
        data = json.load(f)
    
    return (
        np.array(data['train_indices']),
        np.array(data['val_indices']),
        np.array(data['test_indices'])
    )


def save_split_indices(train_indices: np.ndarray,
                       val_indices: np.ndarray,
                       test_indices: np.ndarray,
                       output_path: Path,
                       metadata: dict):
    """
    Save split indices to JSON file for reuse.
    """
    data = {
        'train_indices': train_indices.tolist(),
        'val_indices': val_indices.tolist(),
        'test_indices': test_indices.tolist(),
        'metadata': metadata
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\n✓ Split indices saved to: {output_path.name}")
    print("  (Can be reused for other datasets with the same samples)")


def prompt_use_existing_splits(output_dir: Path) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Check for existing split indices and prompt user to reuse them.
    
    Returns:
        Split indices if user chooses to reuse, None otherwise
    """
    existing_files = find_split_index_files(output_dir)
    
    if not existing_files:
        return None
    
    print("\n" + "=" * 80)
    print("EXISTING SPLIT INDICES FOUND")
    print("=" * 80)
    print("\nFound split index files from previous runs:")
    for i, file in enumerate(existing_files, 1):
        print(f"  [{i}] {file.name}")
    print(f"  [0] Create new split indices")
    print("-" * 80)
    
    choice = input("\nReuse existing split indices? (0 for new): ").strip()
    
    try:
        choice_num = int(choice)
        if choice_num == 0:
            return None
        elif 1 <= choice_num <= len(existing_files):
            selected_file = existing_files[choice_num - 1]
            print(f"\n✓ Loading split indices from: {selected_file.name}")
            
            # Load and display metadata
            with open(selected_file, 'r') as f:
                data = json.load(f)
            
            if 'metadata' in data:
                print("\nSplit metadata:")
                for key, value in data['metadata'].items():
                    print(f"  {key}: {value}")
            
            train_idx, val_idx, test_idx = load_split_indices(selected_file)
            print(f"\n  Train samples: {len(train_idx)}")
            print(f"  Val samples: {len(val_idx)}")
            print(f"  Test samples: {len(test_idx)}")
            
            return (train_idx, val_idx, test_idx)
    except (ValueError, KeyError, FileNotFoundError) as e:
        print(f"\n❌ Error loading split indices: {e}")
        print("Creating new split indices instead...")
        return None
    
    return None


def prompt_save_split_indices(output_dir: Path, filename_prefix: str) -> bool:
    """
    Ask user if they want to save split indices for reuse.
    
    Returns:
        True if user wants to save, False otherwise
    """
    print("\n" + "=" * 80)
    print("SAVE SPLIT INDICES?")
    print("=" * 80)
    print("\nSaving split indices allows you to reuse the exact same")
    print("train/val/test splits for other datasets.")
    print("\nThis ensures consistent splits across multiple preprocessing runs,")
    print("so only labels2 (target domain) varies between datasets.")
    
    default_name = f"{filename_prefix}_split_indices.json"
    print(f"\nDefault filename: {default_name}")
    
    save = input("\nSave split indices? (Y/n): ").strip().lower()
    return save in ['', 'y', 'yes']


def get_stratify_choice() -> str:
    """
    Get stratification choice from user.
    
    Returns:
        "labels1" or "labels2"
    """
    print("\n" + "=" * 80)
    print("STRATIFICATION CONFIGURATION")
    print("=" * 80)
    print("\nStratification options:")
    print("  [1] Stratify on labels1 (cancer_binary: benign/malignant)")
    print("  [2] Stratify on labels2 (target_domain: target/other)")
    print("\nDefault: Stratify on labels1 (cancer_binary)")
    
    choice = input("\nSelect stratification method (1/2): ").strip()
    
    if choice == '2':
        return "labels2"
    else:
        return "labels1"


def get_output_directory() -> Path:
    """
    Get output directory from user.
    
    Returns:
        Path object for output directory
    """
    print("\n" + "=" * 80)
    print("OUTPUT CONFIGURATION")
    print("=" * 80)
    
    default_dir = Path("data/preprocessed")
    print(f"\nDefault output directory: {default_dir}")
    
    use_default = input("Use default output directory? (Y/n): ").strip().lower()
    
    if use_default in ['', 'y', 'yes']:
        output_dir = default_dir
    else:
        custom_path = input("Enter custom output directory path: ").strip()
        output_dir = Path(custom_path)
    
    # Create directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n✓ Output directory: {output_dir.absolute()}")
    
    return output_dir


def get_output_filename(target_domain: str, output_dir: Path) -> str:
    """
    Get custom output filename prefix from user with overwrite protection.
    
    Args:
        target_domain: Target domain name for default filename
        output_dir: Output directory path
        
    Returns:
        Filename prefix (without _train/_val/_test suffix)
    """
    # Create default filename from target domain
    default_prefix = target_domain.lower().replace(" ", "_").replace("-", "_")
    
    print("\n" + "=" * 80)
    print("OUTPUT FILENAME CONFIGURATION")
    print("=" * 80)
    print(f"\nDefault filename prefix: {default_prefix}")
    print(f"  Will create: {default_prefix}_train.npz")
    print(f"               {default_prefix}_val.npz")
    print(f"               {default_prefix}_test.npz")
    
    use_default = input("\nUse default filename? (Y/n): ").strip().lower()
    
    if use_default in ['', 'y', 'yes']:
        filename_prefix = default_prefix
    else:
        while True:
            custom_prefix = input("Enter custom filename prefix: ").strip()
            
            # Sanitize the custom prefix
            filename_prefix = custom_prefix.lower().replace(" ", "_").replace("-", "_")
            
            if filename_prefix:
                print(f"\nSanitized prefix: {filename_prefix}")
                print(f"  Will create: {filename_prefix}_train.npz")
                print(f"               {filename_prefix}_val.npz")
                print(f"               {filename_prefix}_test.npz")
                
                confirm = input("\nConfirm this filename? (Y/n): ").strip().lower()
                if confirm in ['', 'y', 'yes']:
                    break
            else:
                print("ERROR: Filename prefix cannot be empty.")
    
    # Check for existing files
    train_path = output_dir / f"{filename_prefix}_train.npz"
    val_path = output_dir / f"{filename_prefix}_val.npz"
    test_path = output_dir / f"{filename_prefix}_test.npz"
    metadata_path = output_dir / f"{filename_prefix}_metadata.txt"
    
    existing_files = []
    if train_path.exists():
        existing_files.append(train_path.name)
    if val_path.exists():
        existing_files.append(val_path.name)
    if test_path.exists():
        existing_files.append(test_path.name)
    if metadata_path.exists():
        existing_files.append(metadata_path.name)
    
    if existing_files:
        print("\n" + "!" * 80)
        print("WARNING: The following files already exist:")
        for file in existing_files:
            print(f"  - {file}")
        print("!" * 80)
        
        overwrite = input("\nOverwrite existing files? (y/N): ").strip().lower()
        if overwrite not in ['y', 'yes']:
            print("\nPlease choose a different filename.")
            return get_output_filename(target_domain, output_dir)
        else:
            print("\n⚠ Existing files will be overwritten.")
    
    print(f"\n✓ Output filename prefix: {filename_prefix}")
    return filename_prefix


def confirm_preprocessing(target_domain: str, train_size: float, val_size: float, 
                         test_size: float, stratify_on: str, output_dir: Path, 
                         filename_prefix: str) -> bool:
    """
    Display preprocessing configuration and ask for confirmation.
    
    Returns:
        True if user confirms, False otherwise
    """
    print("\n" + "=" * 80)
    print("PREPROCESSING CONFIGURATION SUMMARY")
    print("=" * 80)
    print(f"\n  Target Domain:    {target_domain}")
    print(f"  Train Split:      {train_size * 100:.1f}%")
    print(f"  Val Split:        {val_size * 100:.1f}%")
    print(f"  Test Split:       {test_size * 100:.1f}%")
    print(f"  Stratify On:      {stratify_on}")
    print(f"  Output Directory: {output_dir.absolute()}")
    print(f"  Output Files:     {filename_prefix}_train.npz")
    print(f"                    {filename_prefix}_val.npz")
    print(f"                    {filename_prefix}_test.npz")
    print(f"\n  Label Configuration:")
    print(f"    labels1: cancer_binary (0=benign, 1=malignant)")
    print(f"    labels2: target_domain indicator (1={target_domain}, 0=other)")
    print("\n" + "=" * 80)
    
    confirm = input("\nProceed with preprocessing? (Y/n): ").strip().lower()
    return confirm in ['', 'y', 'yes']


def save_preprocessed_datasets(train_ds, val_ds, test_ds, metadata: dict, 
                              output_dir: Path, filename_prefix: str):
    """
    Save preprocessed datasets to .npz files compatible with experiment config framework.
    
    Args:
        train_ds: Training CustomImageDataset
        val_ds: Validation CustomImageDataset
        test_ds: Test CustomImageDataset
        metadata: Preprocessing metadata dictionary
        output_dir: Output directory path
        filename_prefix: Custom filename prefix for output files
    """
    print("\n" + "=" * 80)
    print("SAVING PREPROCESSED DATASETS")
    print("=" * 80)
    
    # Save train dataset
    train_path = output_dir / f"{filename_prefix}_train.npz"
    print(f"\n[1/3] Saving training set to: {train_path.name}")
    np.savez_compressed(
        train_path,
        images=train_ds.images,
        labels1=train_ds.sub_domain_labels,  # Actually contains cancer_binary (primary task)
        labels2=train_ds.cancer_binary_labels  # Actually contains target_domain indicator (adversarial task)
    )
    print(f"      ✓ Saved {len(train_ds)} samples")
    
    # Save validation dataset
    val_path = output_dir / f"{filename_prefix}_val.npz"
    print(f"\n[2/3] Saving validation set to: {val_path.name}")
    np.savez_compressed(
        val_path,
        images=val_ds.images,
        labels1=val_ds.sub_domain_labels,  # Actually contains cancer_binary (primary task)
        labels2=val_ds.cancer_binary_labels  # Actually contains target_domain indicator (adversarial task)
    )
    print(f"      ✓ Saved {len(val_ds)} samples")
    
    # Save test dataset
    test_path = output_dir / f"{filename_prefix}_test.npz"
    print(f"\n[3/3] Saving test set to: {test_path.name}")
    np.savez_compressed(
        test_path,
        images=test_ds.images,
        labels1=test_ds.sub_domain_labels,  # Actually contains cancer_binary (primary task)
        labels2=test_ds.cancer_binary_labels  # Actually contains target_domain indicator (adversarial task)
    )
    print(f"      ✓ Saved {len(test_ds)} samples")
    
    # Save metadata
    metadata_path = output_dir / f"{filename_prefix}_metadata.txt"
    print(f"\n[4/4] Saving metadata to: {metadata_path.name}")
    with open(metadata_path, 'w') as f:
        f.write("PREPROCESSING METADATA\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Target Domain: {metadata['target_domain']}\n")
        f.write(f"Train Samples: {metadata['train_size']}\n")
        f.write(f"Val Samples: {metadata['val_size']}\n")
        f.write(f"Test Samples: {metadata['test_size']}\n")
        f.write(f"Stratified On: {metadata['stratified_on']}\n")
        f.write(f"Random Seed: {metadata['seed']}\n\n")
        f.write("Label Configuration:\n")
        f.write("  labels1: cancer_binary (0=benign, 1=malignant)\n")
        f.write(f"  labels2: target_domain indicator (1={metadata['target_domain']}, 0=other)\n\n")
        f.write(f"Number of Classes:\n")
        f.write(f"  labels1 (cancer_binary): {metadata['num_classes_labels1']}\n")
        f.write(f"  labels2 (target_domain): {metadata['num_classes_labels2']}\n\n")
        f.write("Train Label Distributions:\n")
        f.write(f"  labels1: {metadata['labels1_distribution_train']}\n")
        f.write(f"  labels2: {metadata['labels2_distribution_train']}\n")
    print(f"      ✓ Metadata saved")
    
    print("\n" + "=" * 80)
    print("PREPROCESSING COMPLETE!")
    print("=" * 80)
    print(f"\nOutput files saved to: {output_dir.absolute()}")
    print(f"\nTo use in experiment configs, reference these files:")
    print(f"  - {train_path.name}")
    print(f"  - {val_path.name}")
    print(f"  - {test_path.name}")
    print("\nExample config snippet:")
    print("  fold_file_paths:")
    print(f"    - {train_path.absolute()}")
    print(f"    - {val_path.absolute()}")
    print("  folds:")
    print("    -")
    print("      train:")
    print("        - 0")
    print("      val:")
    print("        - 1")
    print()


def get_dataset_path() -> str:
    """
    Get dataset path from user with suggestions.
    
    Returns:
        Path to the dataset .npz file
    """
    print("\n" + "=" * 80)
    print("DATASET SELECTION")
    print("=" * 80)
    
    # Suggest common locations
    common_paths = [
        r"data\Multi Cancer\adenocarcinoma_dataset.npz",
        r"data\preprocessed\adenocarcinoma_dataset.npz"
    ]
    
    # Find .npz files in common locations
    available_files = []
    for search_dir in [r"data\Multi Cancer", r"data\preprocessed"]:
        if os.path.exists(search_dir):
            for file in Path(search_dir).glob("*.npz"):
                available_files.append(str(file))
    
    if available_files:
        print("\nFound .npz files in workspace:")
        for idx, file_path in enumerate(available_files, 1):
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"  [{idx}] {file_path} ({file_size_mb:.1f} MB)")
        print(f"  [0] Enter custom path")
        print("-" * 80)
        
        choice = input("\nSelect dataset file (0 for custom): ").strip()
        try:
            choice_num = int(choice)
            if choice_num == 0:
                npz_path = input("Enter custom dataset path: ").strip()
            elif 1 <= choice_num <= len(available_files):
                npz_path = available_files[choice_num - 1]
            else:
                print(f"Invalid choice. Please try again.")
                return get_dataset_path()
        except ValueError:
            print("Invalid input. Please enter a number.")
            return get_dataset_path()
    else:
        print("\nNo .npz files found in common locations.")
        print("Common locations:")
        for path in common_paths:
            print(f"  - {path}")
        print()
        npz_path = input("Enter dataset path: ").strip()
    
    return npz_path


def main():
    """Main execution function."""
    clear_screen()
    print_header()
    
    # Check if dataset path was provided as argument
    if len(sys.argv) > 1:
        npz_path = sys.argv[1]
        print(f"Using dataset from command line: {npz_path}\n")
    else:
        npz_path = get_dataset_path()
    
    # Verify dataset exists
    if not os.path.exists(npz_path):
        print(f"\nERROR: Dataset file not found: {npz_path}")
        print("\nUsage:")
        print(f"  python {sys.argv[0]} [path_to_dataset.npz]")
        sys.exit(1)
    
    print(f"\n✓ Using dataset: {npz_path}\n")
    
    # Get available domains
    print("Loading dataset to extract available domains...")
    domains = get_available_domains(npz_path)
    print(f"✓ Found {len(domains)} cancer domains in dataset\n")
    
    # Display menu and get user selection
    domain_idx = display_menu(domains)
    target_domain = domains[domain_idx]
    
    print(f"\n✓ Selected target domain: {target_domain}")
    
    # Get preprocessing configuration
    train_size, val_size, test_size = get_split_ratios()
    stratify_on = get_stratify_choice()
    output_dir = get_output_directory()
    
    # Check for existing split indices
    reused_indices = prompt_use_existing_splits(output_dir)
    
    filename_prefix = get_output_filename(target_domain, output_dir)
    
    # Confirm before proceeding
    if not confirm_preprocessing(target_domain, train_size, val_size, test_size, 
                                stratify_on, output_dir, filename_prefix):
        print("\nPreprocessing cancelled by user.")
        sys.exit(0)
    
    # Run preprocessing
    print("\n" + "=" * 80)
    print("STARTING PREPROCESSING PIPELINE")
    print("=" * 80)
    print("\nThis may take a few minutes...\n")
    
    try:
        # If reusing indices, we need to manually split the dataset
        if reused_indices is not None:
            print("[Using saved split indices]\n")
            train_idx, val_idx, test_idx = reused_indices
            
            # Import necessary functions
            from cancer_utils import (load_adenocarcinoma_dataset, 
                                     preprocess_for_branched_resnet,
                                     get_branched_resnet_transforms,
                                     create_branched_dataset)
            from collections import Counter
            
            # Load and preprocess
            print("[Step 1/5] Loading adenocarcinoma dataset...")
            images, sub_domain_labels, cancer_binary_labels, domain_text_list, sub_domain_text_list = \
                load_adenocarcinoma_dataset(npz_path)
            
            print(f"\n[Step 2/5] Preprocessing for branched ResNet (target_domain='{target_domain}')...")
            images_processed, labels1, labels2 = preprocess_for_branched_resnet(
                images=images,
                cancer_binary_labels=cancer_binary_labels,
                domain_text_list=domain_text_list,
                target_domain=target_domain,
                normalize=True
            )
            
            print("\n[Step 3/5] Creating image transforms...")
            transform = get_branched_resnet_transforms()
            
            print("\n[Step 4/5] Creating CustomImageDataset...")
            full_dataset = create_branched_dataset(
                images=images_processed,
                labels1=labels1,
                labels2=labels2,
                transform=transform
            )
            
            print(f"\n[Step 5/5] Applying saved split indices...")
            # Create split datasets using saved indices
            from cancer_utils import CustomImageDataset
            
            train_ds = CustomImageDataset(
                images=full_dataset.images[train_idx],
                sub_domain_labels=full_dataset.sub_domain_labels[train_idx],
                cancer_binary_labels=full_dataset.cancer_binary_labels[train_idx],
                transform=transform,
                branched_mode=True
            )
            
            val_ds = CustomImageDataset(
                images=full_dataset.images[val_idx],
                sub_domain_labels=full_dataset.sub_domain_labels[val_idx],
                cancer_binary_labels=full_dataset.cancer_binary_labels[val_idx],
                transform=transform,
                branched_mode=True
            )
            
            test_ds = CustomImageDataset(
                images=full_dataset.images[test_idx],
                sub_domain_labels=full_dataset.sub_domain_labels[test_idx],
                cancer_binary_labels=full_dataset.cancer_binary_labels[test_idx],
                transform=transform,
                branched_mode=True
            )
            
            print(f"  Train: {len(train_ds)} samples")
            print(f"  Val:   {len(val_ds)} samples")
            print(f"  Test:  {len(test_ds)} samples")
            
            # Create metadata
            train_domains = [domain_text_list[i] for i in train_idx]
            val_domains = [domain_text_list[i] for i in val_idx]
            test_domains = [domain_text_list[i] for i in test_idx]
            
            metadata = {
                'target_domain': target_domain,
                'train_size': len(train_ds),
                'val_size': len(val_ds),
                'test_size': len(test_ds),
                'train_domains': train_domains,
                'val_domains': val_domains,
                'test_domains': test_domains,
                'num_classes_labels1': int(np.max(labels1)) + 1,
                'num_classes_labels2': int(np.max(labels2)) + 1,
                'labels1_distribution_train': dict(Counter(train_ds.sub_domain_labels)),
                'labels2_distribution_train': dict(Counter(train_ds.cancer_binary_labels)),
                'stratified_on': stratify_on,
                'seed': 42,
                'used_saved_indices': True
            }
        else:
            # Normal preprocessing with new split
            train_ds, val_ds, test_ds, metadata = cancer_preprocess(
                npz_path=npz_path,
                target_domain=target_domain,
                train_size=train_size,
                val_size=val_size,
                test_size=test_size,
                stratify_on=stratify_on,
                normalize=True,
                seed=42
            )
            
            # Get the indices that were used for splitting
            # We need to extract them from the cancer_preprocess result
            # For now, we'll compute them again with the same seed
            from cancer_utils import load_adenocarcinoma_dataset
            images, _, cancer_binary_labels, domain_text_list, _ = load_adenocarcinoma_dataset(npz_path)
            
            # Compute indices using same stratification
            from cancer_utils import preprocess_for_branched_resnet
            _, labels1, labels2 = preprocess_for_branched_resnet(
                images=images,
                cancer_binary_labels=cancer_binary_labels,
                domain_text_list=domain_text_list,
                target_domain=target_domain,
                normalize=False
            )
            
            # Re-compute split indices
            rng = np.random.default_rng(42)
            N = len(labels1)
            strat_labels = labels1 if stratify_on == "labels1" else labels2
            
            label_to_indices = {}
            for idx, label in enumerate(strat_labels):
                if label not in label_to_indices:
                    label_to_indices[label] = []
                label_to_indices[label].append(idx)
            
            train_indices = []
            val_indices = []
            test_indices = []
            
            for label, indices in label_to_indices.items():
                indices = np.array(indices)
                rng.shuffle(indices)
                n = len(indices)
                n_train = int(n * train_size)
                n_val = int(n * val_size)
                train_indices.extend(indices[:n_train])
                val_indices.extend(indices[n_train:n_train + n_val])
                test_indices.extend(indices[n_train + n_val:])
            
            train_idx = np.array(train_indices)
            val_idx = np.array(val_indices)
            test_idx = np.array(test_indices)
        
        # Save preprocessed datasets
        save_preprocessed_datasets(
            train_ds, val_ds, test_ds, metadata, 
            output_dir, filename_prefix
        )
        
        # Ask if user wants to save indices (only for new splits, not reused ones)
        if reused_indices is None:
            if prompt_save_split_indices(output_dir, filename_prefix):
                indices_path = output_dir / f"{filename_prefix}_split_indices.json"
                indices_metadata = {
                    'source_dataset': npz_path,
                    'target_domain': target_domain,
                    'train_size': train_size,
                    'val_size': val_size,
                    'test_size': test_size,
                    'stratify_on': stratify_on,
                    'seed': 42,
                    'num_train': len(train_idx),
                    'num_val': len(val_idx),
                    'num_test': len(test_idx)
                }
                save_split_indices(train_idx, val_idx, test_idx, indices_path, indices_metadata)
        
        # Ask if user wants to preprocess another domain
        print("\n" + "=" * 80)
        another = input("\nPreprocess another target domain? (y/N): ").strip().lower()
        if another in ['y', 'yes']:
            print("\n")
            main()  # Restart the process
        else:
            print("\nThank you for using the Multi-Cancer Dataset Preprocessor!")
            print("=" * 80)
    
    except KeyboardInterrupt:
        print("\n\nPreprocessing interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nERROR during preprocessing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
