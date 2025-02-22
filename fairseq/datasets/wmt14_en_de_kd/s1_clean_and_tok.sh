#!/bin/bash
data_path=datasets/wmt14_en_de_kd/data
base_path=datasets/spm_tools
lang_dir=en-de

for split in "train" "valid" "test"; do
    for src in "de" "en"; do
        bash ${base_path}/s1_clean_and_tok.sh $src ${data_path}/${split}.${lang_dir}.${src}.txt ${data_path}/${split}.${lang_dir}.${src}.tok
    done
done
