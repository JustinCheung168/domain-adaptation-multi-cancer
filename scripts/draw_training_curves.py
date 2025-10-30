#!/usr/bin/env python3

import argparse
import os

import pandas as pd
import matplotlib.pyplot as plt


def draw_training_curves(csv_path, output_dir):
    # Read the CSV
    df = pd.read_csv(csv_path)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Plot training and eval loss together
    plt.figure()
    plt.plot(df['epoch'], df['train_loss'], label='Train Loss')
    plt.plot(df['epoch'], df['eval_loss'], label='Eval Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Evaluation Loss')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'loss_curve.png'))
    plt.close()

    # Metrics to plot individually
    metrics = [
        'learning_rate',
        'grad_norm',
        'eval_accuracy_branch1',
        'eval_precision_branch1',
        'eval_recall_branch1',
        'eval_f1_branch1',
        'eval_accuracy_branch2',
        'eval_precision_branch2',
        'eval_recall_branch2',
        'eval_f1_branch2',
        'eval_lambda',
    ]

    for metric in metrics:
        if metric in df.columns:
            plt.figure()
            plt.plot(df['epoch'], df[metric])
            plt.xlabel('Epoch')
            plt.ylabel(metric.replace('_', ' ').title())
            plt.title(f'{metric.replace("_", " ").title()} Over Epochs')
            plt.savefig(os.path.join(output_dir, f'{metric}_curve.png'))
            plt.close()

    print(f"Plots saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Train a domain adaptation model, given a config file. Evaluation will be attempted after training as well.")

    parser.add_argument("training_curves_csv_file", type=str, help="Path to CSV file containing data needed to draw training curves. This would be named `training_curves.csv` after using `run_training.py`")
    parser.add_argument("output_dir", type=str, help="Path to directory to put the training curve plots into. This directory doesn't need to already exist.")
    args = parser.parse_args()

    draw_training_curves(args.training_curves_csv_file, args.output_dir)

if __name__ == "__main__":
    main()
