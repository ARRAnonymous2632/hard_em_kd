#!/bin/bash

ckpt_path=${1:-'checkpoints/ckpt_llm_nat/checkpoints_wmt14ende_raw_at_big/avg_ckpt_.pt'}
input=${2:-'data-bin/wmt14_en_de_raw_at_student_6_6_revised'}
device=${3:-"0"}

if [  -f $ckpt_path  ]; then
    avg_ckpt_path=$ckpt_path
    ckpt_par_path=`dirname $ckpt_path`
    tmp_dir_for_eval=$ckpt_par_path/for_eval/$dataset_${decoding_method}_${postfix}/
elif [ -d $ckpt_path ]; then
    avg_ckpt_path=$ckpt_path/avg_ckpt_${dataset}.pt
    python3 -W ignore scripts/average_checkpoints.py --inputs $ckpt_path --output $avg_ckpt_path --num-best-checkpoints 5
    tmp_dir_for_eval=${ckpt_path}/for_eval/${dataset}_${decoding_method}_${postfix}/
fi

tmp_path=$tmp_dir_for_eval
# tmp_path="z_outputs/final/at_xsum"
mkdir -p $tmp_path

# rm $tmp_path/*
dset_name=`basename $input`
output_file_name=${dset_name}_loss_values.log

average_checkpoint_path=$avg_ckpt_path

CUDA_VISIBLE_DEVICES=$device python fairseq_cli/interactive_validation_loss.py data-bin/wmt14.en-de_raw --input ${input} \
    --gen-subset train --user-dir fs_plugins --task translation_rouge \
    --source-lang en --criterion label_smoothed_cross_entropy_mod \
    --target-lang de --max-tokens 8192 \
    --beam 5 \
    --user-dir fs_plugins --skip-invalid-size-inputs-valid-test \
    --remove-bpe --path ${average_checkpoint_path} > $tmp_path/$output_file_name
