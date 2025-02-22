#!/bin/bash

spm_model=$1
finput=$2
foutput=$3
path=$4
nproc=${5:-64}
echo $nproc
mkdir -p $path

split -n l/${nproc} --numeric-suffixes=1 ${finput} ${path}/_TMP_

for part in ${path}/_TMP_*; do
    echo Start encoding $part...
    spm_encode \
        --model=$spm_model \
        --input=$part \
        --output=$part.OUT &
done
wait

cat ${path}/_TMP_*.OUT > ${foutput}
echo removing 

rm ${path}/*
# rm -rf ${path}
