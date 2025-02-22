#!/bin/bash

data_path=datasets/iwslt/kd_data
output_file=glat_iwslt_kd.out

# grep ^H $data_path/$output_file | cut -f 3 > $data_path/$output_file.H.1
# grep ^T $data_path/$output_file | cut -f 2 > $data_path/$output_file.T.1
# grep ^S $data_path/$output_file | cut -f 2 > $data_path/$output_file.S.1
 
# detokenize
cat $data_path/$output_file.H.1 | mosesdecoder/scripts/tokenizer/detokenizer.perl -l en > $data_path/$output_file.H
cat $data_path/$output_file.T.1 | mosesdecoder/scripts/tokenizer/detokenizer.perl -l en > $data_path/$output_file.T
cat $data_path/$output_file.S.1 | mosesdecoder/scripts/tokenizer/detokenizer.perl -l de > $data_path/$output_file.S
