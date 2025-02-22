#!/bin/bash
data_path=datasets/wmt14_en_de_kd/data
base_path=datasets/spm_tools
bin_path=data-bin/wmt14_en_de_kd_spm
lang=en-de

echo "4. Fairseq Preprocess"
rm -rf $bin_path
python3 fairseq_cli/preprocess.py \
    --source-lang en --target-lang de \
    --trainpref $data_path/train.$lang.spm \
    --validpref $data_path/valid.$lang.spm \
    --testpref $data_path/test.$lang.spm \
    --destdir $bin_path \
    --workers 12 --joined-dictionary \
    --srcdict $data_path/$lang.dict.txt
