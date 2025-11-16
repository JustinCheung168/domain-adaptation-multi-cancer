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


def confirm_preprocessing(target_domain: str, train_size: float, val_size: float, 
                         test_size: float, stratify_on: str, output_dir: Path) -> bool:
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
    print(f"\n  Label Configuration:")
    print(f"    labels1: cancer_binary (0=benign, 1=malignant)")
    print(f"    labels2: target_domain indicator (1={target_domain}, 0=other)")
    print("\n" + "=" * 80)
    
    confirm = input("\nProceed with preprocessing? (Y/n): ").strip().lower()
    return confirm in ['', 'y', 'yes']


def save_preprocessed_datasets(train_ds, val_ds, test_ds, metadata: dict, 
                              output_dir: Path, target_domain: str):
    """
    Save preprocessed datasets to .npz files compatible with experiment config framework.
    
    Args:
        train_ds: Training CustomImageDataset
        val_ds: Validation CustomImageDataset
        test_ds: Test CustomImageDataset
        metadata: Preprocessing metadata dictionary
        output_dir: Output directory path
        target_domain: Target domain name for filename
    """
    # Create sanitized filename prefix from target domain
    domain_prefix = target_domain.lower().replace(" ", "_").replace("-", "_")
    
    print("\n" + "=" * 80)
    print("SAVING PREPROCESSED DATASETS")
    print("=" * 80)
    
    # Save train dataset
    train_path = output_dir / f"{domain_prefix}_train.npz"
    print(f"\n[1/3] Saving training set to: {train_path.name}")
    np.savez_compressed(
        train_path,
        images=train_ds.images,
        labels1=train_ds.sub_domain_labels,  # cancer_binary
        labels2=train_ds.cancer_binary_labels  # target_domain indicator
    )
    print(f"      ✓ Saved {len(train_ds)} samples")
    
    # Save validation dataset
    val_path = output_dir / f"{domain_prefix}_val.npz"
    print(f"\n[2/3] Saving validation set to: {val_path.name}")
    np.savez_compressed(
        val_path,
        images=val_ds.images,
        labels1=val_ds.sub_domain_labels,
        labels2=val_ds.cancer_binary_labels
    )
    print(f"      ✓ Saved {len(val_ds)} samples")
    
    # Save test dataset
    test_path = output_dir / f"{domain_prefix}_test.npz"
    print(f"\n[3/3] Saving test set to: {test_path.name}")
    np.savez_compressed(
        test_path,
        images=test_ds.images,
        labels1=test_ds.sub_domain_labels,
        labels2=test_ds.cancer_binary_labels
    )
    print(f"      ✓ Saved {len(test_ds)} samples")
    
    # Save metadata
    metadata_path = output_dir / f"{domain_prefix}_metadata.txt"
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


def main():
    """Main execution function."""
    # Default dataset path
    default_npz_path = r"data\Multi Cancer\adenocarcinoma_dataset.npz"
    
    clear_screen()
    print_header()
    
    # Check if dataset path was provided as argument
    if len(sys.argv) > 1:
        npz_path = sys.argv[1]
    else:
        print(f"Default dataset path: {default_npz_path}")
        use_default = input("Use default dataset path? (Y/n): ").strip().lower()
        
        if use_default in ['', 'y', 'yes']:
            npz_path = default_npz_path
        else:
            npz_path = input("Enter dataset path: ").strip()
    
    # Verify dataset exists
    if not os.path.exists(npz_path):
        print(f"\nERROR: Dataset file not found: {npz_path}")
        print("\nUsage:")
        print(f"  python {sys.argv[0]} [path_to_adenocarcinoma_dataset.npz]")
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
    
    # Confirm before proceeding
    if not confirm_preprocessing(target_domain, train_size, val_size, test_size, 
                                stratify_on, output_dir):
        print("\nPreprocessing cancelled by user.")
        sys.exit(0)
    
    # Run preprocessing
    print("\n" + "=" * 80)
    print("STARTING PREPROCESSING PIPELINE")
    print("=" * 80)
    print("\nThis may take a few minutes...\n")
    
    try:
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
        
        # Save preprocessed datasets
        save_preprocessed_datasets(
            train_ds, val_ds, test_ds, metadata, 
            output_dir, target_domain
        )
        
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
