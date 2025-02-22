#!/bin/bash

all_args=("$@")

config=$1
checkpoint_path=$2

extra=("${all_args[@]:2}")



config_name=`basename $config`
config_name="${config_name%.*}"
exp_name="${config_name}"
if [ ! -z $post_fix ]; then
    exp_name+="_$post_fix"
fi
echo exp name: $exp_name
echo extra commands ${extra[@]}


# set checkpoint_path=checkpoints/ckpt_${project}/checkpoints_${exp_name} if is null
if [ -z $checkpoint_path ]; then
    checkpoint_path="checkpoints/ckpt_${project}/checkpoints_${exp_name}"
fi

# if checkpoint_path is a file, then set checkpoint_path to the parent directory of the file
if [ -f $checkpoint_path ]; then
    checkpoint_path_file=$checkpoint_path
    checkpoint_path=`dirname $checkpoint_path`
fi

mkdir -p $checkpoint_path

# exp_log_path='cmlm_diff'
# mkdir -p $exp_log_path

# config_command="srun --job-name $config_name --time 1-0:00:00  --output $exp_log_path/${config_name}.log --gres=gpu:a100:${num_gpus} "
config_command=""
config_command+=`python3 my_scripts/config_to_command.py --config $config ${extra[@]} --debug `
config_command+=" --save-dir ${checkpoint_path} "
config_command+=" --log-file ${checkpoint_path}/${exp_name}.log "

# checkpoint_path_file is not none, add load checkpoint option (for fairseq) to the config_command
if [ ! -z $checkpoint_path_file ]; then
    config_command+=" --restore-file ${checkpoint_path_file} "
fi

echo $config_command

eval $config_command

sleep 600
