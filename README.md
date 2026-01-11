# VLASH Kinetix Simulation Benchmark

Simulated experiments for real-time execution with VLASH method.

## Installation

```bash
git submodule update --init
uv sync
```

## Evaluation

Pre-trained checkpoint: [mit-han-lab/vlash-kinetix-policy-async5](https://huggingface.co/mit-han-lab/vlash-kinetix-policy-async5)

Evaluate with different real-time control methods (oracle, vlash, rtc, ...):

```bash
# Single GPU
uv run src/eval_flow.py --run-path <checkpoint_path> --output-dir eval_outputs

# Multi-GPU parallel (recommended)
bash scripts/run_parallel.sh <checkpoint_path> eval_outputs 8
```

Plot results:
```bash
uv run scripts/plot_paper_figure.py --input-file eval_outputs/results.csv --output-file eval_outputs/paper_figure6.png
```

## Training

Train flow policy with async delay augmentation:

```bash
bash scripts/train.sh <data_path> <async_interval> <output_dir>
# Example: bash scripts/train.sh /path/to/data 5 ./output
```

## Pre-trained Assets

- Expert checkpoints & data: `gs://rtc-assets/expert/`
- BC policies: `gs://rtc-assets/bc/`
