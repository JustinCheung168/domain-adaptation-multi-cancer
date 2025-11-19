#!/usr/bin/env python3
"""
Draw confusion matrix heatmaps from evaluation results.

This script provides an interactive terminal GUI to select and process
test_metrics.csv files from evaluation results directories and creates
confusion matrix visualizations for Branch 1 (cancer classification).
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def extract_confusion_matrix_from_csv(csv_path: str) -> Tuple[Optional[np.ndarray], Optional[dict]]:
    """
    Extract confusion matrix data from test_metrics.csv file.
    
    Args:
        csv_path: Path to test_metrics.csv file
        
    Returns:
        Tuple of (confusion_matrix, metrics_dict) or (None, None) if data not available
    """
    df = pd.read_csv(csv_path)
    
    # Check if confusion matrix columns exist
    required_cm_cols = [
        'eval_cm_branch1_actual0_pred0',
        'eval_cm_branch1_actual0_pred1',
        'eval_cm_branch1_actual1_pred0',
        'eval_cm_branch1_actual1_pred1'
    ]
    
    if not all(col in df.columns for col in required_cm_cols):
        return None, None
    
    # Extract confusion matrix values for Branch 1 (cancer classification)
    cm = np.array([
        [df['eval_cm_branch1_actual0_pred0'].item(), df['eval_cm_branch1_actual0_pred1'].item()],
        [df['eval_cm_branch1_actual1_pred0'].item(), df['eval_cm_branch1_actual1_pred1'].item()]
    ])
    
    # Extract key metrics
    metrics = {
        'accuracy': df['eval_accuracy_branch1'].item(),
        'precision': df['eval_precision_branch1'].item(),
        'recall': df['eval_recall_branch1'].item(),
        'f1': df['eval_f1_branch1'].item(),
    }
    
    return cm, metrics


def plot_confusion_matrix(cm: np.ndarray, metrics: dict, title: str, 
                         output_path: str, class_names: List[str] = None):
    """
    Plot confusion matrix as a heatmap.
    
    Args:
        cm: Confusion matrix (2x2 numpy array)
        metrics: Dictionary of metrics to display
        title: Plot title
        output_path: Path to save the figure
        class_names: List of class names for labels
    """
    if class_names is None:
        class_names = ['Benign', 'Malignant']
    
    FONTSIZE_TITLE = 16
    FONTSIZE_LABEL = 14
    FONTSIZE_TICK = 12
    FONTSIZE_ANNOT = 14
    FIGSIZE = (8, 7)
    
    # Create figure
    fig, ax = plt.subplots(figsize=FIGSIZE)
    
    # Create heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        linewidths=0.5,
        annot_kws={"size": FONTSIZE_ANNOT},
        ax=ax
    )
    
    # Set labels
    ax.set_xlabel('Predicted Label', fontsize=FONTSIZE_LABEL, labelpad=10)
    ax.set_ylabel('True Label', fontsize=FONTSIZE_LABEL, labelpad=10)
    ax.set_title(title, fontsize=FONTSIZE_TITLE, pad=15)
    
    # Rotate tick labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center', fontsize=FONTSIZE_TICK)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha='right', fontsize=FONTSIZE_TICK)
    
    # Add metrics text box
    metrics_text = (
        f"Accuracy: {metrics['accuracy']:.4f}\n"
        f"Precision: {metrics['precision']:.4f}\n"
        f"Recall: {metrics['recall']:.4f}\n"
        f"F1 Score: {metrics['f1']:.4f}"
    )
    
    ax.text(
        0.02, 0.98, metrics_text,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def find_all_evaluation_dirs(results_root: str) -> dict[str, List[Tuple[str, str]]]:
    """
    Find all evaluation directories grouped by result directory.
    
    Args:
        results_root: Root directory to search
        
    Returns:
        Dictionary mapping result_dir_name to list of (csv_path, eval_dir_name) tuples
    """
    results_path = Path(results_root)
    results_dict = {}
    
    # Search for all test_metrics*.csv files
    for result_dir in results_path.iterdir():
        if not result_dir.is_dir():
            continue
        
        # Skip non-result directories
        if not result_dir.name.startswith('dann_'):
            continue
        
        eval_dirs = []
        # Look for evaluation_* subdirectories
        for subdir in sorted(result_dir.iterdir()):
            if subdir.is_dir() and subdir.name.startswith('evaluation_'):
                # Look for any CSV file that starts with test_metrics
                csv_files = list(subdir.glob('test_metrics*.csv'))
                
                if csv_files:
                    # Use the first matching CSV file
                    csv_path = csv_files[0]
                    
                    # Check if CSV has confusion matrix data
                    try:
                        df = pd.read_csv(csv_path)
                        required_cols = ['eval_cm_branch1_actual0_pred0', 'eval_cm_branch1_actual0_pred1',
                                       'eval_cm_branch1_actual1_pred0', 'eval_cm_branch1_actual1_pred1']
                        if all(col in df.columns for col in required_cols):
                            eval_dirs.append((str(csv_path), subdir.name))
                    except Exception:
                        pass  # Skip invalid CSVs
        
        if eval_dirs:
            results_dict[result_dir.name] = eval_dirs
    
    return results_dict


def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header():
    """Print the application header."""
    print("=" * 80)
    print(" " * 20 + "CONFUSION MATRIX GENERATOR")
    print("=" * 80)
    print()


def display_result_dir_menu(results_dict: dict[str, List[Tuple[str, str]]]) -> List[str]:
    """
    Display menu for selecting result directories to process.
    
    Args:
        results_dict: Dictionary of result_dir_name to evaluation directories
        
    Returns:
        List of selected result directory names
    """
    result_dirs = sorted(results_dict.keys())
    
    print("\nAVAILABLE TEST RESULTS:")
    print("-" * 80)
    
    for idx, result_dir in enumerate(result_dirs, 1):
        num_evals = len(results_dict[result_dir])
        eval_text = f"({num_evals} evaluation{'s' if num_evals > 1 else ''})"
        print(f"  [{idx}] {result_dir} {eval_text}")
    
    print(f"\n  [A] Process ALL results")
    print(f"  [0] Exit")
    print("-" * 80)
    
    while True:
        try:
            choice = input("\nSelect result directories (comma-separated numbers, 'A' for all, or 0 to exit): ").strip()
            
            if choice == '0':
                print("\nExiting. Goodbye!")
                sys.exit(0)
            
            if choice.upper() == 'A':
                return result_dirs
            
            # Parse comma-separated numbers
            selections = []
            for part in choice.split(','):
                part = part.strip()
                if '-' in part:
                    # Handle ranges like "1-5"
                    start, end = map(int, part.split('-'))
                    selections.extend(range(start - 1, end))
                else:
                    selections.append(int(part) - 1)
            
            # Validate selections
            if all(0 <= s < len(result_dirs) for s in selections):
                return [result_dirs[i] for i in sorted(set(selections))]
            else:
                print(f"Invalid selection. Please enter numbers between 1 and {len(result_dirs)}.")
        except ValueError:
            print("Invalid input. Please enter comma-separated numbers, ranges (e.g., 1-5), or 'A'.")
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Exiting...")
            sys.exit(0)


def display_evaluation_menu(result_dir_name: str, eval_dirs: List[Tuple[str, str]]) -> List[int]:
    """
    Display menu for selecting evaluation runs within a result directory.
    
    Args:
        result_dir_name: Name of the result directory
        eval_dirs: List of (csv_path, eval_dir_name) tuples
        
    Returns:
        List of selected evaluation indices
    """
    print(f"\n{result_dir_name}")
    print("-" * 80)
    print(f"Found {len(eval_dirs)} evaluation run(s) with valid confusion matrix data:")
    print()
    
    for idx, (csv_path, eval_dir_name) in enumerate(eval_dirs, 1):
        # Extract timestamp from evaluation directory name
        print(f"  [{idx}] {eval_dir_name}")
    
    if len(eval_dirs) == 1:
        print(f"\n  [Enter] Use the only available evaluation")
    else:
        print(f"\n  [A] Use ALL evaluations")
    print(f"  [0] Skip this result directory")
    print("-" * 80)
    
    while True:
        try:
            if len(eval_dirs) == 1:
                choice = input("\nPress Enter to continue or 0 to skip: ").strip()
                if choice == '' or choice == '1':
                    return [0]
                elif choice == '0':
                    return []
            else:
                choice = input(f"\nSelect evaluations (comma-separated, 'A' for all, or 0 to skip): ").strip()
                
                if choice == '0':
                    return []
                
                if choice.upper() == 'A':
                    return list(range(len(eval_dirs)))
                
                # Parse comma-separated numbers
                selections = []
                for part in choice.split(','):
                    part = part.strip()
                    if '-' in part:
                        start, end = map(int, part.split('-'))
                        selections.extend(range(start - 1, end))
                    else:
                        selections.append(int(part) - 1)
                
                # Validate selections
                if all(0 <= s < len(eval_dirs) for s in selections):
                    return sorted(set(selections))
                else:
                    print(f"Invalid selection. Please enter numbers between 1 and {len(eval_dirs)}.")
        except ValueError:
            print("Invalid input. Please enter numbers, 'A', or 0.")
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Exiting...")
            sys.exit(0)


def get_output_directory() -> Path:
    """
    Get output directory from user.
    
    Returns:
        Path object for output directory
    """
    print("\n" + "=" * 80)
    print("OUTPUT CONFIGURATION")
    print("=" * 80)
    
    default_dir = Path("results/figs/confusion_matrices")
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


def confirm_processing(selected_files: List[Tuple[str, str]], output_dir: Path) -> bool:
    """
    Display processing configuration and ask for confirmation.
    
    Returns:
        True if user confirms, False otherwise
    """
    print("\n" + "=" * 80)
    print("PROCESSING SUMMARY")
    print("=" * 80)
    print(f"\n  Results to process: {len(selected_files)}")
    for _, result_dir in selected_files:
        print(f"    - {result_dir}")
    print(f"\n  Output directory: {output_dir.absolute()}")
    print("\n" + "=" * 80)
    
    confirm = input("\nProceed with generating confusion matrices? (Y/n): ").strip().lower()
    return confirm in ['', 'y', 'yes']


def process_result_directory(csv_path: str, result_dir_name: str, eval_dir_name: str, output_dir: str) -> bool:
    """
    Process a single results directory and create confusion matrix.
    
    Args:
        csv_path: Path to test_metrics.csv file
        result_dir_name: Name of the result directory
        eval_dir_name: Name of the evaluation directory
        output_dir: Directory to save output figures
        
    Returns:
        True if successful, False if skipped
    """
    print(f"\n  Processing: {result_dir_name} / {eval_dir_name}")
    print(f"    Metrics file: {csv_path}")
    
    # Extract confusion matrix and metrics
    cm, metrics = extract_confusion_matrix_from_csv(csv_path)
    
    if cm is None or metrics is None:
        print(f"    ⚠️  Skipping: No confusion matrix data found in CSV")
        return False
    
    # Generate title and output filename
    model_name = result_dir_name.replace('dann_test_', '').replace('dann_train_val_', '').replace('_results', '')
    
    # Include evaluation directory in filename if there are multiple
    eval_suffix = f"_{eval_dir_name}" if not eval_dir_name.startswith('evaluation_0_') else ""
    
    title = f"{model_name.replace('_', ' ').title()} - Confusion Matrix"
    if eval_suffix:
        title += f"\n{eval_dir_name}"
    
    output_filename = f"confusion_matrix_{model_name}{eval_suffix}.png"
    output_path = os.path.join(output_dir, output_filename)
    
    # Plot and save
    plot_confusion_matrix(cm, metrics, title, output_path)
    return True


def process_all_split_results(results_root: str = 'results', output_dir: str = 'results/figs/confusion_matrices'):
    """
    Process all result directories (non-interactive mode).
    
    Args:
        results_root: Root directory containing results
        output_dir: Directory to save output figures
    """
    results_path = Path(results_root)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("CONFUSION MATRIX GENERATOR")
    print("=" * 80)
    print(f"\nResults directory: {results_path.absolute()}")
    print(f"Output directory: {Path(output_dir).absolute()}")
    
    # Find all evaluation directories
    results_dict = find_all_evaluation_dirs(results_root)
    
    if not results_dict:
        print("\n❌ No test_metrics.csv files with confusion matrix data found!")
        return
    
    total_evals = sum(len(evals) for evals in results_dict.values())
    print(f"\nFound {len(results_dict)} result directories with {total_evals} evaluation run(s):")
    for result_dir, eval_dirs in sorted(results_dict.items()):
        print(f"  - {result_dir} ({len(eval_dirs)} evaluation{'s' if len(eval_dirs) > 1 else ''})")
    
    print("\n" + "=" * 80)
    print("PROCESSING RESULTS")
    print("=" * 80)
    
    # Process all evaluations in all directories
    success_count = 0
    skipped_count = 0
    
    for result_dir_name in sorted(results_dict.keys()):
        print(f"\n{result_dir_name}:")
        for csv_path, eval_dir_name in results_dict[result_dir_name]:
            if process_result_directory(csv_path, result_dir_name, eval_dir_name, output_dir):
                success_count += 1
            else:
                skipped_count += 1
    
    print("\n" + "=" * 80)
    print(f"COMPLETE! Generated {success_count} confusion matrices")
    if skipped_count > 0:
        print(f"Skipped {skipped_count} files (no confusion matrix data)")
    print("=" * 80)
    print(f"\nAll figures saved to: {Path(output_dir).absolute()}")


def interactive_mode(results_root: str = 'results'):
    """
    Run interactive terminal GUI mode.
    
    Args:
        results_root: Root directory containing results
    """
    clear_screen()
    print_header()
    
    # Find all available evaluation directories
    print("Searching for test results...")
    results_dict = find_all_evaluation_dirs(results_root)
    
    if not results_dict:
        print(f"\n❌ No test_metrics.csv files with confusion matrix data found in {results_root}!")
        print("\nMake sure you have evaluation results with the following structure:")
        print("  results/")
        print("    dann_test_*_results/")
        print("      evaluation_*/")
        print("        test_metrics.csv")
        sys.exit(1)
    
    total_evals = sum(len(evals) for evals in results_dict.values())
    print(f"✓ Found {len(results_dict)} result directories with {total_evals} evaluation run(s)\n")
    
    # Display result directory selection menu
    selected_result_dirs = display_result_dir_menu(results_dict)
    
    if not selected_result_dirs:
        print("\n❌ No result directories selected!")
        sys.exit(0)
    
    # For each selected result directory, let user choose evaluation runs
    selected_files = []
    
    for result_dir_name in selected_result_dirs:
        eval_dirs = results_dict[result_dir_name]
        
        if len(eval_dirs) == 1:
            # Automatically use the single evaluation
            selected_files.append((eval_dirs[0][0], result_dir_name, eval_dirs[0][1]))
        else:
            # Let user choose which evaluation runs to process
            selected_eval_indices = display_evaluation_menu(result_dir_name, eval_dirs)
            
            if not selected_eval_indices:
                print(f"  Skipped {result_dir_name}")
                continue
            
            for idx in selected_eval_indices:
                csv_path, eval_dir_name = eval_dirs[idx]
                selected_files.append((csv_path, result_dir_name, eval_dir_name))
    
    if not selected_files:
        print("\n❌ No evaluations selected!")
        sys.exit(0)
    
    # Get output directory
    output_dir = get_output_directory()
    
    # Confirm before processing
    print("\n" + "=" * 80)
    print("PROCESSING SUMMARY")
    print("=" * 80)
    print(f"\n  Total evaluations to process: {len(selected_files)}")
    for csv_path, result_dir, eval_dir in selected_files:
        print(f"    - {result_dir} / {eval_dir}")
    print(f"\n  Output directory: {output_dir.absolute()}")
    print("\n" + "=" * 80)
    
    confirm = input("\nProceed with generating confusion matrices? (Y/n): ").strip().lower()
    if confirm not in ['', 'y', 'yes']:
        print("\nProcessing cancelled by user.")
        sys.exit(0)
    
    # Process selected files
    print("\n" + "=" * 80)
    print("GENERATING CONFUSION MATRICES")
    print("=" * 80)
    
    success_count = 0
    skipped_count = 0
    for csv_path, result_dir_name, eval_dir_name in selected_files:
        if process_result_directory(csv_path, result_dir_name, eval_dir_name, str(output_dir)):
            success_count += 1
        else:
            skipped_count += 1
    
    print("\n" + "=" * 80)
    print(f"COMPLETE! Generated {success_count} confusion matrices")
    if skipped_count > 0:
        print(f"Skipped {skipped_count} files (no confusion matrix data)")
    print("=" * 80)
    print(f"\nAll figures saved to: {output_dir.absolute()}")
    
    # Ask if user wants to process more
    print("\n" + "=" * 80)
    another = input("\nGenerate more confusion matrices? (y/N): ").strip().lower()
    if another in ['y', 'yes']:
        print("\n")
        interactive_mode(results_root)
    else:
        print("\nThank you for using the Confusion Matrix Generator!")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Generate confusion matrix heatmaps from evaluation results.",
        epilog="If no arguments are provided, interactive mode will be launched."
    )
    parser.add_argument(
        '--results-root',
        type=str,
        default='results',
        help='Root directory containing results (default: results)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory to save confusion matrix figures (default: results/figs/confusion_matrices)'
    )
    parser.add_argument(
        '--csv-path',
        type=str,
        default=None,
        help='Process a specific test_metrics.csv file (provide full path)'
    )
    parser.add_argument(
        '--result-name',
        type=str,
        default=None,
        help='Custom result name when using --csv-path'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Process all test_metrics.csv files found in results-root (non-interactive)'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Force interactive mode (default if no other options specified)'
    )
    
    args = parser.parse_args()
    
    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = 'results/figs/confusion_matrices'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if we should run in interactive mode
    if args.interactive or (not args.csv_path and not args.all):
        # Interactive mode
        interactive_mode(args.results_root)
    elif args.csv_path:
        # Process single CSV file
        if not os.path.exists(args.csv_path):
            print(f"❌ File not found: {args.csv_path}")
            sys.exit(1)
        
        csv_path_obj = Path(args.csv_path)
        eval_dir_name = csv_path_obj.parent.name
        result_dir_name = args.result_name if args.result_name else csv_path_obj.parent.parent.name
        
        print("=" * 80)
        print("CONFUSION MATRIX GENERATOR")
        print("=" * 80)
        process_result_directory(args.csv_path, result_dir_name, eval_dir_name, output_dir)
        print("\n" + "=" * 80)
        print("COMPLETE!")
        print("=" * 80)
        print(f"\nFigure saved to: {Path(output_dir).absolute()}")
    elif args.all:
        # Process all automatically
        process_all_split_results(args.results_root, output_dir)


if __name__ == "__main__":
    main()
