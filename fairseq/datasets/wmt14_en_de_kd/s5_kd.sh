#!/bin/bash

# bash datasets/wmt14_de_en_kd/s5_kd.sh checkpoints/ckpt_llm_nat/checkpoints_wmt14deen_kd_ctc data-bin/wmt14_en_de_kd_spm 4 0


ckpt_path=$1
dataset_bin=${2-"data-bin/wmt14_en_de_kd_spm"}
num_gpus=${3:-8}
gpt_start_id=${4:-0}
gpu_end_id=$((gpt_start_id+num_gpus-1))

# avg checkoints if it is a dir
if [  -f $ckpt_path  ]; then
    avg_ckpt_path=$ckpt_path
    ckpt_par_path=`dirname $ckpt_path`
elif [ -d $ckpt_path ]; then
    avg_ckpt_path=$ckpt_path/avg_ckpt_${dataset}.pt
    python3 -W ignore scripts/average_checkpoints.py --inputs $ckpt_path --output $avg_ckpt_path --num-best-checkpoints 5
fi

# KD with multiple GPUs 
tmp_dir_for_eval=${ckpt_path}/for_eval/
mkdir -p $tmp_dir_for_eval 

    
echo KD $src "-->" $trg 
for rank in $(seq $gpt_start_id $gpu_end_id); do
    CUDA_VISIBLE_DEVICES=$rank python3 fairseq_cli/generate.py  $dataset_bin  --path $avg_ckpt_path \
        --task translation_lev   --gen-subset train  --source-lang en \
        --target-lang de --batch-size 100 \
        --user-dir fs_plugins \
        --iter-decode-max-iter 0 --iter-decode-eos-penalty 0 --beam 1 \
        --remove-bpe sentencepiece \
        --distributed-world-size ${num_gpus} --distributed-rank $((rank-gpt_start_id))  \
        > $tmp_dir_for_eval/kd.out.shard$((rank-gpt_start_id)) &
        # > $tmp_dir_for_eval/${src}2${trg}.out &
done    
wait

cat $tmp_dir_for_eval/kd.out.shard* > $tmp_dir_for_eval/kd.out.shard.out
# rm $tmp_dir_for_eval/kd.out.shard*


output_file=$tmp_dir_for_eval/kd.out.shard.out
 
 # extract 
grep ^H $output_file | cut -f 3 > $output_file.H.1
grep ^T $output_file | cut -f 2 > $output_file.T.1
grep ^S $output_file | cut -f 2 > $output_file.S.1
 
# detokenize
cat $output_file.H.1 | mosesdecoder/scripts/tokenizer/detokenizer.perl -l en > $output_file.H.2
cat $output_file.T.1 | mosesdecoder/scripts/tokenizer/detokenizer.perl -l en > $output_file.T.2
cat $output_file.S.1 | mosesdecoder/scripts/tokenizer/detokenizer.perl -l de > $output_file.S.2

# checkpoints/ckpt_nat/checkpoints_iwslt_glat_new/checkpoint_best.pt
# checkpoints/ckpt_nat/checkpoints_iwslt_nat_new.checkpoint_best.pt