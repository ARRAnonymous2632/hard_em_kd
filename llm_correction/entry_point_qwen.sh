#!/bin/bash


prompt=$1
hypos=$2
targets=$3
sources=$4
EXPERIMENT_DIR=$5
start_idx=${6:-0}
end_idx=${7:-1}
use_targets=${8:-0}
num_gpus=${9:-1}
seed=${10:-6547}
dt=${11:-0}

eval python qwen_batch_inference.py \
    --prompt ${prompt} --hypos ${hypos} --targets ${targets} --sources ${sources}\
    --output_dir ${EXPERIMENT_DIR} --start_idx ${start_idx} --end_idx ${end_idx} --use_targets ${use_targets} \
    --num_gpus ${num_gpus} --seed ${seed} --dt ${dt}
