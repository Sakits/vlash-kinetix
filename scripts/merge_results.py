#!/usr/bin/env python
"""Merge parallel evaluation results."""

import argparse
import pathlib
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    
    input_dir = pathlib.Path(args.input_dir)
    parts = sorted(input_dir.glob("results_part_*.csv"))
    
    if not parts:
        print("No result parts found")
        return
    
    df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    df.to_csv(args.output, index=False)
    print(f"Merged {len(parts)} parts -> {args.output} ({len(df)} rows)")


if __name__ == "__main__":
    main()


