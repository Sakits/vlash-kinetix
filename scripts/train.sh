#!/bin/bash
# Train flow policy with async delay augmentation

export NCCL_NET_MERGE_LEVEL=LOC
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN

export CUDA_VISIBLE_DEVICES=0,1,2,3

uv run src/train_flow.py \
    --config.run-path /path/to/rtc/ \
    --config.async-interval 5 \
    --config.output-dir /path/to/vlash_async_5_policy
