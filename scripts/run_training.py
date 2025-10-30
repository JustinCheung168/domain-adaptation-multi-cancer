#!/usr/bin/env python3

import argparse
from domain_adaptation_ct.learn.training_funcs import run_training_from_config_file

def main():
    parser = argparse.ArgumentParser(description="Train a domain adaptation model, given a config file. Evaluation will be attempted after training as well.")

    parser.add_argument("config_file", type=str, help="Path to config file to run training for.")
    args = parser.parse_args()

    run_training_from_config_file(args.config_file)

if __name__ == "__main__":
    main()
