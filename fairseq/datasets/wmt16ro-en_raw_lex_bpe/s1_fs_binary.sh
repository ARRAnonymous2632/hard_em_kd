#!/bin/bash
data_path=datasets/wmtro-en_raw/data
base_path=datasets/spm_tools
bin_path=data-bin/wmtro-en_raw
lang=ro-en

echo "4. Fairseq Preprocess"
rm -rf $bin_path
python3 fairseq_cli/preprocess.py \
    --source-lang ro --target-lang en \
    --trainpref $data_path/train.$lang.tok \
    --validpref $data_path/valid.$lang.tok \
    --testpref $data_path/test.$lang.tok \
    --destdir $bin_path \
    --workers 12 --joined-dictionary \
    --srcdict data-bin/wmt16ro-en_raw_lex_bpe/dict.ro.txt
