#!/usr/bin/env python3
"""
Interactive Multi-Cancer Dataset Creator

This script provides an interactive terminal GUI to:
1. Load or create formatted_data.npz from raw image folders
2. Select specific cancer domains to include
3. Create custom datasets with chosen domains
4. Save datasets with custom names
"""

import os
import sys
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np
from PIL import Image
import cv2

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header():
    """Print the application header."""
    print("=" * 80)
    print(" " * 20 + "MULTI-CANCER DATASET CREATOR")
    print("=" * 80)
    print()


def load_images_from_folders(data_path: str, target_size=(224, 224), max_per_folder=None) -> dict:
    """
    Load images from each top-level subfolder under data_path.
    
    Returns: Dict[str, Dict[str, List[np.ndarray]]]
        {
          top_level_folder: {
              subfolder_path: [images ...]
          },
          ...
        }
    """
    data = {}
    if not os.path.isdir(data_path):
        raise FileNotFoundError(f"Data path does not exist: {data_path}")

    allowed_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    
    print(f"Loading images from: {data_path}")
    
    for top_folder in os.listdir(data_path):
        top_folder_path = os.path.join(data_path, top_folder)
        if not os.path.isdir(top_folder_path):
            continue

        print(f"  Processing domain: {top_folder}...")
        subdict = {}

        for root, _dirs, files in os.walk(top_folder_path):
            img_files = [f for f in files if f.lower().endswith(allowed_ext)]
            if len(img_files) == 0:
                continue

            images = []
            loaded_count = 0
            for f in img_files:
                file_path = os.path.join(root, f)
                try:
                    with Image.open(file_path) as img:
                        img = img.convert('RGB')
                        if target_size is not None:
                            img = img.resize(target_size, Image.BILINEAR)
                        images.append(np.array(img))
                        loaded_count += 1
                except Exception as e:
                    print(f"    Warning: failed to load {file_path}: {e}")
                    continue

                if isinstance(max_per_folder, int) and max_per_folder > 0 and loaded_count >= max_per_folder:
                    break

            if len(images) == 0:
                continue

            rel_sub = os.path.relpath(root, top_folder_path).replace('\\', '/')
            if rel_sub == '.':
                rel_sub = '(root)'

            subdict[rel_sub] = images

        if len(subdict) == 0:
            print(f"    Warning: no image files found in {top_folder_path}; skipping.")
            continue

        total_images = sum(len(imgs) for imgs in subdict.values())
        print(f"    Loaded {total_images} images from {len(subdict)} subfolder(s)")
        data[top_folder] = subdict
    
    return data


def create_formatted_data(data: dict, benign_keys: List[str]) -> dict:
    """
    Create formatted dataset from loaded images.
    
    Returns dictionary with:
        - images: (N, H, W, 3) numpy array
        - labels_text: (N,) string array of class names
        - labels_id: (N,) int64 array of class IDs
        - binary_labels: (N,) int64 array (0=benign, 1=malignant)
        - domain_text: (N,) string array of domain names
        - domain_id: (N,) int64 array of domain IDs
    """
    print("\nCreating formatted dataset...")
    
    images = []
    labels = []
    binary_labels = []
    domain_labels = []

    for top, subdict in data.items():
        for key, imgs in subdict.items():
            is_benign = any(bkey in key for bkey in benign_keys)
            bin_lab = 0 if is_benign else 1
            images.extend(imgs)
            labels.extend([key] * len(imgs))
            binary_labels.extend([bin_lab] * len(imgs))
            domain_labels.extend([top] * len(imgs))

    # Convert to numpy arrays
    images_np = np.stack(images)
    labels_np = np.array(labels)
    binary_labels_np = np.array(binary_labels, dtype=np.int64)
    domain_labels_np = np.array(domain_labels)

    # Create integer-encoded domain IDs
    unique_domains = sorted(list(set(domain_labels)))
    domain_to_id = {d: i for i, d in enumerate(unique_domains)}
    domain_ids_np = np.array([domain_to_id[d] for d in domain_labels], dtype=np.int64)

    # Create class IDs
    unique_classes = sorted(np.unique(labels_np))
    class_to_id = {c: i for i, c in enumerate(unique_classes)}
    class_ids_np = np.array([class_to_id[c] for c in labels_np], dtype=np.int64)

    formatted_data = {
        "images": images_np,
        "labels_text": labels_np,
        "labels_id": class_ids_np,
        "binary_labels": binary_labels_np,
        "domain_text": domain_labels_np,
        "domain_id": domain_ids_np
    }
    
    print(f"  Total samples: {len(images_np)}")
    print(f"  Image shape: {images_np.shape}")
    print(f"  Unique domains: {len(unique_domains)}")
    print(f"  Unique classes: {len(unique_classes)}")
    print(f"  Benign samples: {np.sum(binary_labels_np == 0)}")
    print(f"  Malignant samples: {np.sum(binary_labels_np == 1)}")
    
    return formatted_data


def display_data_source_menu() -> Tuple[str, Optional[str]]:
    """
    Display menu for choosing data source.
    
    Returns:
        Tuple of (choice, formatted_data_path)
        choice: 'load' or 'create'
        formatted_data_path: path if loading, None if creating
    """
    print("\n" + "=" * 80)
    print("DATA SOURCE")
    print("=" * 80)
    print("\nHow would you like to obtain the formatted dataset?")
    print("  [1] Load existing formatted_data.npz file")
    print("  [2] Create new formatted_data.npz from raw images")
    print("  [0] Exit")
    print("-" * 80)
    
    while True:
        try:
            choice = input("\nSelect option (1/2): ").strip()
            
            if choice == '0':
                print("\nExiting. Goodbye!")
                sys.exit(0)
            
            if choice == '1':
                # Load existing file
                default_path = "data/Multi Cancer/formatted_data.npz"
                print(f"\nDefault path: {default_path}")
                use_default = input("Use default path? (Y/n): ").strip().lower()
                
                if use_default in ['', 'y', 'yes']:
                    path = default_path
                else:
                    path = input("Enter path to formatted_data.npz: ").strip()
                
                if not os.path.exists(path):
                    print(f"❌ File not found: {path}")
                    continue
                
                return 'load', path
            
            elif choice == '2':
                return 'create', None
            
            else:
                print("Invalid choice. Please enter 1 or 2.")
        
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Exiting...")
            sys.exit(0)


def get_raw_data_path() -> str:
    """Get path to raw image data."""
    print("\n" + "=" * 80)
    print("RAW IMAGE DATA LOCATION")
    print("=" * 80)
    
    default_path = "data/Multi Cancer"
    print(f"\nDefault path: {default_path}")
    use_default = input("Use default path? (Y/n): ").strip().lower()
    
    if use_default in ['', 'y', 'yes']:
        return default_path
    else:
        return input("Enter path to raw image folder: ").strip()


def get_benign_keywords() -> List[str]:
    """Get keywords for identifying benign samples."""
    print("\n" + "=" * 80)
    print("BENIGN SAMPLE IDENTIFICATION")
    print("=" * 80)
    
    default_keys = ['benign', 'normal', 'pab', 'sfi', 'bnt']
    print(f"\nDefault benign keywords: {', '.join(default_keys)}")
    print("(Folders containing these keywords will be labeled as benign)")
    
    use_default = input("\nUse default keywords? (Y/n): ").strip().lower()
    
    if use_default in ['', 'y', 'yes']:
        return default_keys
    else:
        custom = input("Enter comma-separated keywords: ").strip()
        return [k.strip() for k in custom.split(',') if k.strip()]


def display_domain_selection_menu(available_domains: List[str]) -> List[str]:
    """
    Display menu for selecting domains to include.
    
    Returns:
        List of selected domain names
    """
    print("\n" + "=" * 80)
    print("DOMAIN SELECTION")
    print("=" * 80)
    print(f"\nAvailable domains ({len(available_domains)}):")
    print("-" * 80)
    
    for idx, domain in enumerate(available_domains, 1):
        print(f"  [{idx}] {domain}")
    
    print(f"\n  [A] Include ALL domains")
    print(f"  [0] Exit")
    print("-" * 80)
    
    while True:
        try:
            choice = input("\nSelect domains (comma-separated numbers, 'A' for all): ").strip()
            
            if choice == '0':
                print("\nExiting. Goodbye!")
                sys.exit(0)
            
            if choice.upper() == 'A':
                return available_domains
            
            # Parse comma-separated numbers
            selections = []
            for part in choice.split(','):
                part = part.strip()
                if '-' in part:
                    # Handle ranges like "1-3"
                    start, end = map(int, part.split('-'))
                    selections.extend(range(start - 1, end))
                else:
                    selections.append(int(part) - 1)
            
            # Validate selections
            if all(0 <= s < len(available_domains) for s in selections):
                return [available_domains[i] for i in sorted(set(selections))]
            else:
                print(f"Invalid selection. Please enter numbers between 1 and {len(available_domains)}.")
        
        except ValueError:
            print("Invalid input. Please enter comma-separated numbers, ranges (e.g., 1-3), or 'A'.")
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Exiting...")
            sys.exit(0)


def get_output_name(selected_domains: List[str]) -> str:
    """Get output filename for the dataset."""
    print("\n" + "=" * 80)
    print("OUTPUT FILENAME")
    print("=" * 80)
    
    # Generate suggested name from domains
    if len(selected_domains) == 1:
        suggested = selected_domains[0].lower().replace(' ', '_').replace('-', '_')
    else:
        suggested = "multi_domain"
    
    suggested += "_dataset"
    
    print(f"\nSuggested filename: {suggested}.npz")
    use_suggested = input("Use suggested filename? (Y/n): ").strip().lower()
    
    if use_suggested in ['', 'y', 'yes']:
        return suggested
    else:
        custom = input("Enter custom filename (without .npz): ").strip()
        return custom if custom else suggested


def get_output_directory() -> Path:
    """Get output directory."""
    print("\n" + "=" * 80)
    print("OUTPUT DIRECTORY")
    print("=" * 80)
    
    default_dir = Path("data/preprocessed")
    print(f"\nDefault directory: {default_dir}")
    
    use_default = input("Use default directory? (Y/n): ").strip().lower()
    
    if use_default in ['', 'y', 'yes']:
        output_dir = default_dir
    else:
        custom_path = input("Enter custom directory path: ").strip()
        output_dir = Path(custom_path)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n✓ Output directory: {output_dir.absolute()}")
    
    return output_dir


def display_summary(formatted_data: dict, selected_domains: List[str], 
                   output_path: Path) -> bool:
    """
    Display summary and ask for confirmation.
    
    Returns:
        True if user confirms, False otherwise
    """
    # Filter data to selected domains
    domain_mask = np.isin(formatted_data['domain_text'], selected_domains)
    n_samples = np.sum(domain_mask)
    n_benign = np.sum(formatted_data['binary_labels'][domain_mask] == 0)
    n_malignant = np.sum(formatted_data['binary_labels'][domain_mask] == 1)
    
    # Count samples per domain
    domain_counts = {}
    for domain in selected_domains:
        mask = formatted_data['domain_text'] == domain
        domain_counts[domain] = np.sum(mask)
    
    print("\n" + "=" * 80)
    print("DATASET SUMMARY")
    print("=" * 80)
    print(f"\nSelected Domains: {len(selected_domains)}")
    for domain, count in domain_counts.items():
        print(f"  - {domain}: {count} samples")
    
    print(f"\nTotal Samples: {n_samples}")
    print(f"  Benign: {n_benign} ({100*n_benign/n_samples:.1f}%)")
    print(f"  Malignant: {n_malignant} ({100*n_malignant/n_samples:.1f}%)")
    
    print(f"\nOutput File: {output_path}")
    print("\n" + "=" * 80)
    
    confirm = input("\nProceed with creating dataset? (Y/n): ").strip().lower()
    return confirm in ['', 'y', 'yes']


def save_dataset(formatted_data: dict, selected_domains: List[str], output_path: Path):
    """Save the filtered dataset."""
    print("\n" + "=" * 80)
    print("CREATING DATASET")
    print("=" * 80)
    
    # Filter data to selected domains
    domain_mask = np.isin(formatted_data['domain_text'], selected_domains)
    
    filtered_data = {
        'images': formatted_data['images'][domain_mask],
        'cancer_binary': formatted_data['binary_labels'][domain_mask],  # Binary cancer labels (0=benign, 1=malignant)
        'sub_domain': formatted_data['labels_id'][domain_mask],         # Original label IDs (class encoding)
        'domain_text': formatted_data['domain_text'][domain_mask],
        'sub_domain_text': formatted_data['labels_text'][domain_mask],
    }
    
    print(f"\nSaving dataset to: {output_path}")
    np.savez_compressed(output_path, **filtered_data)
    
    print(f"✓ Dataset saved successfully!")
    print(f"\nDataset contents:")
    for key, value in filtered_data.items():
        if hasattr(value, 'shape'):
            print(f"  {key}: {value.shape}")
        else:
            print(f"  {key}: {type(value)}")
    
    # Save metadata
    metadata_path = output_path.parent / f"{output_path.stem}_metadata.txt"
    with open(metadata_path, 'w') as f:
        f.write("DATASET METADATA\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Output File: {output_path.name}\n")
        f.write(f"Total Samples: {len(filtered_data['images'])}\n")
        f.write(f"Benign: {np.sum(filtered_data['cancer_binary'] == 0)}\n")
        f.write(f"Malignant: {np.sum(filtered_data['cancer_binary'] == 1)}\n\n")
        f.write("Selected Domains:\n")
        for domain in selected_domains:
            mask = filtered_data['domain_text'] == domain
            count = np.sum(mask)
            f.write(f"  - {domain}: {count} samples\n")
        f.write("\nLabel Configuration:\n")
        f.write("  cancer_binary: Binary cancer labels (0=benign, 1=malignant)\n")
        f.write("  sub_domain: Original class label IDs\n")
        f.write("  domain_text: Domain names (cancer types)\n")
        f.write("  sub_domain_text: Original class label text\n")
    
    print(f"✓ Metadata saved to: {metadata_path}")


def main():
    """Main execution function."""
    clear_screen()
    print_header()
    
    try:
        # Step 1: Get data source
        choice, formatted_data_path = display_data_source_menu()
        
        if choice == 'load':
            # Load existing formatted_data.npz
            print(f"\nLoading formatted data from: {formatted_data_path}")
            loaded = np.load(formatted_data_path, allow_pickle=True)
            formatted_data = {
                'images': loaded['images'],
                'labels_text': loaded['labels_text'],
                'labels_id': loaded['labels_id'],
                'binary_labels': loaded['binary_labels'],
                'domain_text': loaded['domain_text'],
                'domain_id': loaded['domain_id']
            }
            print("✓ Formatted data loaded successfully!")
            
        else:  # choice == 'create'
            # Create new formatted_data.npz
            raw_data_path = get_raw_data_path()
            
            if not os.path.exists(raw_data_path):
                print(f"\n❌ Path not found: {raw_data_path}")
                sys.exit(1)
            
            benign_keys = get_benign_keywords()
            
            print("\n" + "=" * 80)
            print("LOADING RAW IMAGES")
            print("=" * 80)
            
            data = load_images_from_folders(raw_data_path, target_size=(224, 224))
            
            if not data:
                print("\n❌ No images found!")
                sys.exit(1)
            
            formatted_data = create_formatted_data(data, benign_keys)
            
            # Ask if user wants to save formatted_data
            print("\n" + "=" * 80)
            save_formatted = input("\nSave formatted_data.npz for future use? (Y/n): ").strip().lower()
            
            if save_formatted in ['', 'y', 'yes']:
                formatted_path = Path("data/Multi Cancer/formatted_data.npz")
                formatted_path.parent.mkdir(parents=True, exist_ok=True)
                print(f"Saving to: {formatted_path}")
                np.savez_compressed(formatted_path, **formatted_data)
                print("✓ Formatted data saved!")
        
        # Step 2: Get available domains
        unique_domains = sorted(np.unique(formatted_data['domain_text']).tolist())
        
        # Step 3: Select domains
        selected_domains = display_domain_selection_menu(unique_domains)
        
        if not selected_domains:
            print("\n❌ No domains selected!")
            sys.exit(0)
        
        print(f"\n✓ Selected {len(selected_domains)} domain(s)")
        
        # Step 4: Get output name and directory
        output_name = get_output_name(selected_domains)
        output_dir = get_output_directory()
        output_path = output_dir / f"{output_name}.npz"
        
        # Step 5: Display summary and confirm
        if not display_summary(formatted_data, selected_domains, output_path):
            print("\nDataset creation cancelled by user.")
            sys.exit(0)
        
        # Step 6: Save dataset
        save_dataset(formatted_data, selected_domains, output_path)
        
        print("\n" + "=" * 80)
        print("COMPLETE!")
        print("=" * 80)
        print(f"\nDataset saved to: {output_path.absolute()}")
        
        # Ask if user wants to create another dataset
        print("\n" + "=" * 80)
        another = input("\nCreate another dataset? (y/N): ").strip().lower()
        if another in ['y', 'yes']:
            print("\n")
            main()
        else:
            print("\nThank you for using the Multi-Cancer Dataset Creator!")
            print("=" * 80)
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
