"""
Split preprocessed NK datasets by domain.

This script takes a preprocessed dataset (e.g., colon_nk_test.npz) and splits it into
separate files for each domain (e.g., colon_nk_test_colon.npz, colon_nk_test_breast.npz, 
colon_nk_test_lung.npz).

It uses the original adenocarcinoma_no_kidney.npz and split indices to map samples
back to their original domains.
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple
import argparse


def load_split_indices(split_indices_path: str) -> Dict[str, np.ndarray]:
    """Load split indices from JSON file."""
    with open(split_indices_path, 'r') as f:
        data = json.load(f)
    
    return {
        'train': np.array(data['train_indices']),
        'val': np.array(data['val_indices']),
        'test': np.array(data['test_indices'])
    }


def get_domain_mapping(original_npz_path: str, split_indices: Dict[str, np.ndarray], 
                       split_name: str) -> Dict[str, List[int]]:
    """
    Create a mapping of domain names to local indices in the split dataset.
    
    Args:
        original_npz_path: Path to original adenocarcinoma_no_kidney.npz
        split_indices: Dictionary with 'train', 'val', 'test' indices
        split_name: Which split to process ('train', 'val', or 'test')
    
    Returns:
        Dictionary mapping domain names to list of local indices in the split
    """
    # Load original dataset
    original_data = np.load(original_npz_path)
    domain_text = original_data['domain_text']
    
    # Get the global indices for this split
    global_indices = split_indices[split_name]
    
    # Map domain names to local indices in the split
    domain_to_local_indices = {}
    
    for local_idx, global_idx in enumerate(global_indices):
        domain = domain_text[global_idx]
        if domain not in domain_to_local_indices:
            domain_to_local_indices[domain] = []
        domain_to_local_indices[domain].append(local_idx)
    
    return domain_to_local_indices


def verify_split(preprocessed_npz_path: str, 
                original_npz_path: str,
                split_indices_path: str,
                domain_files: Dict[str, Path],
                split_name: str) -> bool:
    """
    Verify that domain-split datasets are correct.
    
    Args:
        preprocessed_npz_path: Path to original preprocessed dataset
        original_npz_path: Path to original adenocarcinoma_no_kidney.npz
        split_indices_path: Path to split indices JSON file
        domain_files: Dictionary mapping domain names to their output file paths
        split_name: Split name ('train', 'val', or 'test')
    
    Returns:
        True if verification passed, False otherwise
    """
    print()
    print("=" * 80)
    print("VERIFICATION")
    print("=" * 80)
    print()
    
    # Load original data
    preprocessed_data = np.load(preprocessed_npz_path)
    original_data = np.load(original_npz_path)
    split_indices = load_split_indices(split_indices_path)
    
    images_original = preprocessed_data['images']
    labels1_original = preprocessed_data['labels1']
    labels2_original = preprocessed_data['labels2']
    
    domain_text = original_data['domain_text']
    global_indices = split_indices[split_name]
    
    verification_passed = True
    total_samples_check = 0
    
    print("Verification checks:")
    print()
    
    # Check 1: Verify each domain file
    for domain, domain_path in sorted(domain_files.items()):
        domain_data = np.load(domain_path)
        domain_images = domain_data['images']
        domain_labels1 = domain_data['labels1']
        domain_labels2 = domain_data['labels2']
        
        total_samples_check += len(domain_images)
        
        # Get expected indices for this domain
        expected_local_indices = []
        for local_idx, global_idx in enumerate(global_indices):
            if domain_text[global_idx] == domain:
                expected_local_indices.append(local_idx)
        
        # Verify sample count
        if len(domain_images) != len(expected_local_indices):
            print(f"❌ {domain}: Sample count mismatch!")
            print(f"    Expected: {len(expected_local_indices)}, Got: {len(domain_images)}")
            verification_passed = False
            continue
        
        # Verify images match
        images_match = np.array_equal(
            domain_images, 
            images_original[expected_local_indices]
        )
        
        # Verify labels1 match
        labels1_match = np.array_equal(
            domain_labels1,
            labels1_original[expected_local_indices]
        )
        
        # Verify labels2 match
        labels2_match = np.array_equal(
            domain_labels2,
            labels2_original[expected_local_indices]
        )
        
        if images_match and labels1_match and labels2_match:
            print(f"✓ {domain}: All {len(domain_images)} samples verified")
        else:
            print(f"❌ {domain}: Data mismatch detected!")
            if not images_match:
                print(f"    Images do not match")
            if not labels1_match:
                print(f"    Labels1 do not match")
            if not labels2_match:
                print(f"    Labels2 do not match")
            verification_passed = False
    
    print()
    
    # Check 2: Verify total sample count
    if total_samples_check == len(images_original):
        print(f"✓ Total samples: {total_samples_check} (matches original)")
    else:
        print(f"❌ Total samples mismatch!")
        print(f"    Expected: {len(images_original)}, Got: {total_samples_check}")
        verification_passed = False
    
    # Check 3: Verify no duplicates or missing samples
    all_local_indices = []
    for domain, domain_path in domain_files.items():
        domain_data = np.load(domain_path)
        for local_idx, global_idx in enumerate(global_indices):
            if domain_text[global_idx] == domain:
                all_local_indices.append(local_idx)
    
    all_local_indices_set = set(all_local_indices)
    expected_indices_set = set(range(len(images_original)))
    
    if all_local_indices_set == expected_indices_set:
        print(f"✓ No missing or duplicate samples")
    else:
        print(f"❌ Missing or duplicate samples detected!")
        missing = expected_indices_set - all_local_indices_set
        duplicates = len(all_local_indices) - len(all_local_indices_set)
        if missing:
            print(f"    Missing indices: {sorted(missing)[:10]}..." if len(missing) > 10 else f"    Missing indices: {sorted(missing)}")
        if duplicates:
            print(f"    Duplicate count: {duplicates}")
        verification_passed = False
    
    print()
    print("=" * 80)
    
    if verification_passed:
        print("✓✓✓ VERIFICATION PASSED ✓✓✓")
    else:
        print("❌❌❌ VERIFICATION FAILED ❌❌❌")
    
    print("=" * 80)
    print()
    
    return verification_passed


def split_dataset_by_domain(preprocessed_npz_path: str, 
                            original_npz_path: str,
                            split_indices_path: str,
                            output_dir: str = None,
                            verify: bool = True) -> None:
    """
    Split a preprocessed dataset by domain.
    
    Args:
        preprocessed_npz_path: Path to preprocessed dataset (e.g., colon_nk_test.npz)
        original_npz_path: Path to original adenocarcinoma_no_kidney.npz
        split_indices_path: Path to split indices JSON file
        output_dir: Output directory (defaults to same as preprocessed_npz_path)
        verify: If True, verify the split datasets after creation (default: True)
    """
    # Parse file paths
    preprocessed_path = Path(preprocessed_npz_path)
    if output_dir is None:
        output_dir = preprocessed_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract base name and split type from filename
    # e.g., "colon_nk_test.npz" -> base="colon_nk", split_name="test"
    # e.g., "colon_nk_test_reinhard.npz" -> base="colon_nk_reinhard", split_name="test"
    filename = preprocessed_path.stem  # "colon_nk_test" or "colon_nk_test_reinhard"
    parts = filename.split('_')
    
    # Find the split type (train/val/test) in the filename
    split_name = None
    split_idx = None
    for i, part in enumerate(parts):
        if part in ['train', 'val', 'test']:
            split_name = part
            split_idx = i
            break
    
    if split_name is None:
        raise ValueError(f"Could not determine split type from filename: {filename}")
    
    # Base name includes everything except the split type
    # e.g., "breast_nk_test_reinhard" -> "breast_nk_reinhard"
    base_name = '_'.join(parts[:split_idx] + parts[split_idx+1:])
    
    print(f"Processing: {preprocessed_path.name}")
    print(f"  Base name: {base_name}")
    print(f"  Split type: {split_name}")
    print()
    
    # Load preprocessed data
    preprocessed_data = np.load(preprocessed_npz_path)
    images = preprocessed_data['images']
    labels1 = preprocessed_data['labels1']
    labels2 = preprocessed_data['labels2']
    
    print(f"Loaded preprocessed data:")
    print(f"  Images: {images.shape}")
    print(f"  Labels1: {labels1.shape}")
    print(f"  Labels2: {labels2.shape}")
    print()
    
    # Load split indices
    split_indices = load_split_indices(split_indices_path)
    print(f"Loaded split indices from: {split_indices_path}")
    print(f"  Train: {len(split_indices['train'])} samples")
    print(f"  Val: {len(split_indices['val'])} samples")
    print(f"  Test: {len(split_indices['test'])} samples")
    print()
    
    # Get domain mapping
    domain_to_local_indices = get_domain_mapping(
        original_npz_path, split_indices, split_name
    )
    
    print(f"Domain distribution in {split_name} split:")
    for domain, indices in sorted(domain_to_local_indices.items()):
        print(f"  {domain}: {len(indices)} samples")
    print()
    
    # Split and save by domain
    print("Splitting and saving datasets by domain...")
    print("=" * 80)
    
    domain_files = {}  # Track created files for verification
    
    for domain, local_indices in sorted(domain_to_local_indices.items()):
        # Create domain-specific filename
        # e.g., "Breast Cancer" -> "breast", "Colon Cancer" -> "colon"
        domain_short = domain.split()[0].lower()  # "breast", "colon", "lung"
        output_filename = f"{base_name}_{split_name}_{domain_short}.npz"
        output_path = output_dir / output_filename
        
        # Extract samples for this domain
        domain_images = images[local_indices]
        domain_labels1 = labels1[local_indices]
        domain_labels2 = labels2[local_indices]
        
        # Save domain-specific dataset
        np.savez(
            output_path,
            images=domain_images,
            labels1=domain_labels1,
            labels2=domain_labels2
        )
        
        domain_files[domain] = output_path
        
        print(f"✓ Saved: {output_filename}")
        print(f"    Path: {output_path}")
        print(f"    Domain: {domain}")
        print(f"    Samples: {len(local_indices)}")
        print(f"    Images shape: {domain_images.shape}")
        print(f"    Labels1 distribution: {dict(zip(*np.unique(domain_labels1, return_counts=True)))}")
        print(f"    Labels2 distribution: {dict(zip(*np.unique(domain_labels2, return_counts=True)))}")
        print()
    
    print("=" * 80)
    print("✓ Dataset splitting complete!")
    print()
    
    # Verify the split if requested
    if verify:
        verification_passed = verify_split(
            preprocessed_npz_path=preprocessed_npz_path,
            original_npz_path=original_npz_path,
            split_indices_path=split_indices_path,
            domain_files=domain_files,
            split_name=split_name
        )
        
        if not verification_passed:
            print("WARNING: Verification failed! Please check the output files.")
            print()


def main():
    parser = argparse.ArgumentParser(
        description="Split preprocessed NK datasets by domain",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Split colon_nk_test.npz by domain
  python split_dataset_by_domain.py data/preprocessed/colon_nk_test.npz
  
  # Split breast_nk_val.npz with custom split indices
  python split_dataset_by_domain.py data/preprocessed/breast_nk_val.npz \\
      --split-indices data/preprocessed/breast_nk_split_indices.json
  
  # Split and save to custom output directory
  python split_dataset_by_domain.py data/preprocessed/lung_nk_test.npz \\
      --output-dir results/domain_splits/
        """
    )
    
    parser.add_argument(
        'preprocessed_npz',
        type=str,
        help='Path to preprocessed dataset (e.g., colon_nk_test.npz)'
    )
    
    parser.add_argument(
        '--original-npz',
        type=str,
        default='data/preprocessed/adenocarcinoma_no_kidney.npz',
        help='Path to original adenocarcinoma_no_kidney.npz (default: data/preprocessed/adenocarcinoma_no_kidney.npz)'
    )
    
    parser.add_argument(
        '--split-indices',
        type=str,
        default='data/preprocessed/breast_nk_split_indices.json',
        help='Path to split indices JSON file (default: data/preprocessed/breast_nk_split_indices.json)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for split datasets (default: same as input file)'
    )
    
    parser.add_argument(
        '--no-verify',
        action='store_true',
        help='Skip verification after splitting (default: verification enabled)'
    )
    
    args = parser.parse_args()
    
    # Use the provided split indices file (defaults to breast_nk_split_indices.json)
    # This works because all NK datasets share the same split indices
    if not Path(args.split_indices).exists():
        print(f"ERROR: Split indices file not found: {args.split_indices}")
        print("Please specify --split-indices manually")
        return
    
    # Verify files exist
    if not Path(args.preprocessed_npz).exists():
        print(f"ERROR: Preprocessed dataset not found: {args.preprocessed_npz}")
        return
    
    if not Path(args.original_npz).exists():
        print(f"ERROR: Original dataset not found: {args.original_npz}")
        return
    
    if not Path(args.split_indices).exists():
        print(f"ERROR: Split indices file not found: {args.split_indices}")
        return
    
    # Run the splitting
    print("=" * 80)
    print("SPLIT DATASET BY DOMAIN")
    print("=" * 80)
    print()
    
    split_dataset_by_domain(
        preprocessed_npz_path=args.preprocessed_npz,
        original_npz_path=args.original_npz,
        split_indices_path=args.split_indices,
        output_dir=args.output_dir,
        verify=not args.no_verify
    )


if __name__ == '__main__':
    main()
