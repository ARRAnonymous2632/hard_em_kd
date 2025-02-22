
#!/bin/bash


all_args=("$@")
dataset=$1

# mkdir -p data-bin

if [ $dataset == "efd" ]; then
    lang_pair_list=("en-fr" "de-en" )
    base_path=/mnt/bd/chenyang-drive-cloud-xl/data_mglat/wmt_en_de_fr
elif [ $dataset == "efz" ]; then
    dataset_bin="efz_85K"
    lang_pair_list=("en-fr" "en-zh")
    base_path=/mnt/bd/chenyang-drive-cloud-xl/data_mglat/wmt_en_fr_zh
else [ $dataset == "many" ]
    langs="en,de,fr,ro,ru,zh"
    lang_pair_list=("de-en"  "en-fr"  "en-ro" "en-ru" "en-zh")    
    base_path=/mnt/bd/chenyang-drive-cloud-xl/data_mglat/wmt_en_de_fr_ro_ru_zh2

fi

output_path=/mnt/bd/chenyang-drive-cloud-xl/song_$dataset

mkdir -p $output_path

for lang_pair in ${lang_pair_list[@]}; do
    src=${lang_pair:0:2}
    trg=${lang_pair:3:4}
    cat $base_path/rank*/train.$lang_pair.$src > $output_path/train.$lang_pair.$src &
    cat $base_path/rank*/train.$lang_pair.$trg > $output_path/train.$lang_pair.$trg &
done

wait 

echo fair preprocess
for lang_pair in ${lang_pair_list[@]}
do
    src=${lang_pair:0:2}
    trg=${lang_pair:3:4}
    echo processing $src $trg
    train_path=$base_path
    val_test_path=$base_path

    process_path=$config_name/${src}-${trg}
        
    #   src --> trg
    #   trg --> src
    for direction in ${src}-${trg} ${trg}-${src}
    do  
        new_src=${direction:0:2}
        new_trg=${direction:3:4}

        bin_path=$output_path/data-bin/$direction
        rm -rf $bin_path
        python3 fairseq_cli/preprocess.py \
            --source-lang $new_src --target-lang $new_trg \
            --trainpref $output_path/train.$lang_pair \
            --validpref $output_path/valid.$direction \
            --testpref $output_path/valid.$direction \
            --destdir $bin_path \
            --workers 128 --joined-dictionary \
            --srcdict $output_path/dict.txt
    done
done