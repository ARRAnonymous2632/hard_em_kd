ckpt_path=$1
dataset_bin=${2-"data-bin/wmt14.en-de_kd"}
beam_size=${3-"5"}
device_rank=${4-"0"}

# avg checkoints if it is a dir
if [  -f $ckpt_path  ]; then
    avg_ckpt_path=$ckpt_path
    ckpt_par_path=`dirname $ckpt_path`
    tmp_dir_for_eval=${ckpt_par_path}/for_eval
elif [ -d $ckpt_path ]; then
    avg_ckpt_path=$ckpt_path/avg_ckpt_${dataset}.pt
    python3 -W ignore scripts/average_checkpoints.py --inputs $ckpt_path --output $avg_ckpt_path --num-best-checkpoints 5
    tmp_dir_for_eval=${ckpt_path}/for_eval
    ckpt_path=$avg_ckpt_path
fi

mkdir -p $tmp_dir_for_eval
CUDA_VISIBLE_DEVICES=${device_rank} python3 fairseq_cli/generate.py  $dataset_bin  --path $ckpt_path \
    --task translation_rouge  --gen-subset test  --source-lang en \
    --target-lang de --max-tokens 16384 \
    --beam ${beam_size}\
    --user-dir fs_plugins --skip-invalid-size-inputs-valid-test \
    --remove-bpe  --seed 0 > $tmp_dir_for_eval/test.beam${beam_size}.out

output_file=$tmp_dir_for_eval/test.beam${beam_size}.out

grep ^H- $output_file | sort -t'-' -k2,2n | cut -f 3 > $output_file.H.1 &
grep ^T- $output_file | sort -t'-' -k2,2n | cut -f 2 > $output_file.T.1 &
grep ^S- $output_file | sort -t'-' -k2,2n | cut -f 2 > $output_file.S.1 &
 wait

# detokenize
cat $output_file.H.1 | mosesdecoder/scripts/tokenizer/detokenizer.perl -l de > $output_file.H.2 &
cat $output_file.T.1 | mosesdecoder/scripts/tokenizer/detokenizer.perl -l de > $output_file.T.2 &
cat $output_file.S.1 | mosesdecoder/scripts/tokenizer/detokenizer.perl -l en > $output_file.S.2 &

wait

sacrebleu --input $output_file.H.2 -t wmt14/full -l en-de -m bleu chrf -w 4 > $output_file.SACREBLEU.RESULT

CUDA_VISIBLE_DEVICES=${device_rank} comet-score -s $output_file.S.2 -t $output_file.H.2 -r data-bin/wmt14.en-de_kd/test.de.ori.detok > $output_file.COMET-SCORE.RESULTS

echo "Done"