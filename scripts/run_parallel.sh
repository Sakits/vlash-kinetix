#!/bin/bash
# Run evaluation in parallel across multiple GPUs

RUN_PATH=${1:?Usage: $0 <run_path> [output_dir] [num_gpus] [step]}
OUTPUT_DIR=${2:-"eval_outputs"}
NUM_GPUS=${3:-8}
STEP=${4:--1}  # -1 means last checkpoint

mkdir -p "$OUTPUT_DIR"

echo "Parallel eval: $NUM_GPUS GPUs, step=$STEP"
echo "Run path: $RUN_PATH"
echo "Output: $OUTPUT_DIR"

for i in $(seq 0 $((NUM_GPUS-1))); do
    CUDA_VISIBLE_DEVICES=$i PYTHONUNBUFFERED=1 uv run src/eval_flow.py \
        --run-path "$RUN_PATH" \
        --output-dir "$OUTPUT_DIR" \
        --config.step $STEP \
        --parallel-index $i \
        --parallel-total $NUM_GPUS \
        2>&1 | tee "$OUTPUT_DIR/gpu_$i.log" &
done

wait
echo "All jobs completed. Merging results..."
uv run scripts/merge_results.py --input-dir "$OUTPUT_DIR" --output "$OUTPUT_DIR/results.csv"
