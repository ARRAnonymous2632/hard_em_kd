#!/bin/bash
data_path=datasets/wmt14_en_de_kd/data
base_path=datasets/spm_tools
lang=en-de


for split in "train" "valid" "test"; do
    for src in "de" "en"; do
        mkdir -p tmp_${split}_${src}
        bash $base_path/s3_apply_spm.sh $data_path/$lang.sp.model $data_path/$split.$lang.$src.tok \
            $data_path/$split.$lang.spm.$src  tmp_${split}_${src}
    done
done

