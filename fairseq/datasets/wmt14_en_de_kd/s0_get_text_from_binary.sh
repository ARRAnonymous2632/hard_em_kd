#!/bin/bash

data_path=wmt14.en-de_kd
output_path=datasets/wmt14_en_de_kd/data
mkdir -p $output_path
lang_dir=en-de

for split in "train" "valid" "test"; do
    for src in "de" "en"; do
        python scripts/read_binarized.py --dataset-impl mmap  --dict data-bin/${data_path}/dict.${src}.txt --input data-bin/${data_path}/${split}.${lang_dir}.${src} | python my_scripts/remove_bpe.py | mosesdecoder/scripts/tokenizer/detokenizer.perl -l ${src} >  ${output_path}/${split}.${lang_dir}.${src}.txt
    done
done
