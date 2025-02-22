#!/bin/bash
data_path=datasets/wmt14_en_de_kd/data
base_path=datasets/spm_tools
vocab_size=40000
lang=en-de

cat ${data_path}/train.$lang.en.tok ${data_path}/train.$lang.de.tok > ${data_path}/train.$lang.cat.tok


bash $base_path/s2_learn_spm.sh $lang ${data_path}/train.$lang.cat.tok $vocab_size $data_path bpe

cut -f1 $data_path/$lang.sp.vocab | tail -n +4 | sed "s/$/ 100/g" > $data_path/$lang.dict.txt.1
tail -n +2 $data_path/$lang.dict.txt.1 > $data_path/$lang.dict.txt

