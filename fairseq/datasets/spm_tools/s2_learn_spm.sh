lang=$1
finput=$2
vocab_size=$3
path=$4
name=$path/${lang}.sp
spm_model="${5:-unigram}"

spm_train \
    --input="${finput}" \
    --model_prefix="${name}" \
    --vocab_size=$vocab_size \
    --unk_id=3 \
    --bos_id=0 \
    --eos_id=2 \
    --pad_id=1 \
    --model_type=$spm_model \
    --num_threads=20 \
    --character_coverage=0.999999 \
    --input_sentence_size=10000000 \
    --shuffle_input_sentence=true \
    --train_extremely_large_corpus=true \
    --max_sentence_length=2048
    > ${name}.log 2>&1

sed -i -E 's/\.[0-9]+$//g' ${name}.vocab
