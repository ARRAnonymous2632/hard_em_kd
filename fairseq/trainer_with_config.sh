#!/bin/bash

all_args=("$@")
config=$1
project=${2:-test}
project_path=${3:-None}
data_bin=${4:-"None"} 
extra=("${@:5}")
config_name=`basename $config`
config_name="${config_name%.*}"
exp_name="${config_name}"
if [ ! -z $post_fix ]; then
    exp_name+="_$post_fix"
fi


echo exp name: $exp_name
echo extra commands ${extra[@]}

if [ $project_path == "None" ]; then
    checkpoint_path=checkpoints/ckpt_${project}/checkpoints_${exp_name}
else
    checkpoint_path=$project_path
fi

mkdir -p $checkpoint_path

config_command=""
config_command+=`python3 my_scripts/config_to_command_with_local_wandb.py --config $config --data-bin ${data_bin} `
config_command+=" --save-dir ${checkpoint_path} "
config_command+=" --log-file ${checkpoint_path}/${exp_name}.log "
config_command+=" --wandb-project $project "
config_command+=" ${extra[@]} "

echo $config_command

eval $config_command

# sleep 600
