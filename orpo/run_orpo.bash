#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p ./logs

DIRECTORY="qwen_instruct_orpo"
LOG_FILE="${DIRECTORY}.log"

nohup python -m run.train \
    --model_id Qwen/Qwen2-7B-Instruct  \
    --save_dir ./saved_models/qwen_instruct \
    --batch_size 1 \
    --gradient_accumulation_steps 4 \
    --warmup_steps 50 \
    --lr 2e-5 \
    --epoch 40 \
    > ./logs/$LOG_FILE 2>&1 &

echo "Training started. Check logs at ./logs/$LOG_FILE"
