#!/usr/bin/env python3

import argparse
from domain_adaptation_ct.learn.evaluation_funcs import run_evaluation_from_config_file

def main():
    parser = argparse.ArgumentParser(description="Evaluate a domain adaptation model, given a config file.")

    parser.add_argument("config_file", type=str, help="Path to config file to run evaluation for.")
    args = parser.parse_args()

    run_evaluation_from_config_file(args.config_file)

if __name__ == "__main__":
    main()
