#!/usr/bin/env python3
"""
Split test datasets into separate target-only and source-only files.

This script reads test .npz files and splits them based on the labels2 field:
- labels2 == 1: Target domain samples
- labels2 == 0: Source domain samples

Output files are saved with _target_only.npz and _source_only.npz suffixes.
"""

import argparse
import numpy as np
from pathlib import Path


def split_test_dataset(input_path: str, output_dir: str = None):
    """
    Split a test dataset into target-only and source-only files.
    
    Args:
        input_path: Path to the input .npz file
        output_dir: Directory to save output files (defaults to same as input)
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        print(f"❌ File not found: {input_path}")
        return False
    
    print(f"\n{'='*80}")
    print(f"Processing: {input_path.name}")
    print(f"{'='*80}")
    
    # Load the data
    data = np.load(str(input_path))
    images = data['images']
    labels1 = data['labels1']
    labels2 = data['labels2']
    
    print(f"Total samples: {len(images)}")
    print(f"Images shape: {images.shape}")
    
    # Split by domain
    target_mask = labels2 == 1
    source_mask = labels2 == 0
    
    target_count = target_mask.sum()
    source_count = source_mask.sum()
    
    print(f"\nTarget domain samples (labels2=1): {target_count}")
    print(f"Source domain samples (labels2=0): {source_count}")
    
    if target_count == 0:
        print("⚠️  Warning: No target domain samples found!")
    if source_count == 0:
        print("⚠️  Warning: No source domain samples found!")
    
    # Determine output directory
    if output_dir is None:
        output_dir = input_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create output filenames
    base_name = input_path.stem  # Filename without extension
    target_output = output_dir / f"{base_name}_target_only.npz"
    source_output = output_dir / f"{base_name}_source_only.npz"
    
    # Save target-only dataset
    if target_count > 0:
        np.savez_compressed(
            str(target_output),
            images=images[target_mask],
            labels1=labels1[target_mask],
            labels2=labels2[target_mask]
        )
        print(f"\n✓ Saved target-only dataset: {target_output}")
        print(f"  Samples: {target_count}")
    
    # Save source-only dataset
    if source_count > 0:
        np.savez_compressed(
            str(source_output),
            images=images[source_mask],
            labels1=labels1[source_mask],
            labels2=labels2[source_mask]
        )
        print(f"✓ Saved source-only dataset: {source_output}")
        print(f"  Samples: {source_count}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Split test datasets into target-only and source-only files."
    )
    parser.add_argument(
        "input_files",
        nargs="+",
        type=str,
        help="Path(s) to input .npz test files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save output files (defaults to same directory as input)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("TEST DATASET SPLITTER")
    print("="*80)
    print(f"Processing {len(args.input_files)} file(s)...")
    
    success_count = 0
    for input_file in args.input_files:
        if split_test_dataset(input_file, args.output_dir):
            success_count += 1
    
    print("\n" + "="*80)
    print(f"SUMMARY: Successfully processed {success_count}/{len(args.input_files)} files")
    print("="*80)


if __name__ == "__main__":
    main()
