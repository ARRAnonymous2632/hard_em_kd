#!/bin/bash

get_update_freq(){
    local ntk_or_bsz=$1
    local card_num=$2
    local ntk_or_bsz_per_card=$3
    echo ntk_or_bsz=$ntk_or_bsz, ntk_or_bsz_per_card=$ntk_or_bsz_per_card  >&2
    local update=$(echo "scale=2;$ntk_or_bsz/$card_num/$ntk_or_bsz_per_card" | bc)
    local update=$(printf "%.0f\n" $update)
    echo update_freq is caculated as $update  >&2
    echo $update
}

get_ckpt_tsbd(){
    case $1 in 
    local )
        local ckpt=checkpoints/$2
        local tsbd=tensorboard/$2
        ;;
    hdfs )
        # local tsbd=xxxx
        ;;
    mix )
        local ckpt=checkpoints/$2
        # local tsbd=xxxxxx
        ;;
    * )
        exit 0 ;;
    esac
    echo "$ckpt $tsbd"
}

start_inotify(){
    local local_ckpt=$1
    local ckpt=$2
    pkill -9 inotifywait
    mkdir -p ${local_ckpt}
    hadoop fs -mkdir ${ckpt}
    # hadoop fs -get ${ckpt}/* ${local_ckpt}
    # hadoop fs -put ${local_ckpt}/* ${ckpt}
    (inotifywait -m ${local_ckpt} -e close_write -e create -e moved_to |
        while read path action file; do
            if [[ "$file" =~ .*pt$ ]]; then
                echo "| inotifywait --> | checkpoint detected: $file"
                hadoop fs -put -f ${local_ckpt}/$file ${ckpt}/ 
                echo "| inotifywait --> | checkpoint uploaded: $file"
                python3 my_scripts/sync_arnold_util.py ${local_ckpt} ${ckpt}
            fi
        done) 
}

get_cuda_num(){
    if [ "$ARNOLD_DEBUG" = "vscode" ]
    then
        local n=$(python3 -c 'import torch; print(torch.cuda.device_count())')
    else
        local n=$(echo "$ARNOLD_WORKER_NUM*$ARNOLD_WORKER_GPU" | bc)
    fi
    echo Found $n cuda devices >&2
    echo $n
}

download_if_not_exists(){
    if [ -d $1 ]
    then
        echo $1 exists >&2
    else
        local src=
        echo $1 not exists, download from $src >&2
        hadoop fs -get $src
    fi
}


infer_lang_pair(){
    if [[ $1 =~ deen ]]; then
        echo de en
    else
        echo en de
    fi
}


only_master_fetch_ckpt(){
    echo "try to fetch ckpt $1"
    if [[ $1 != 'off' && $1 != 'hf' ]]; then
        python3 fetch_checkpoint.py --idx $1
        # if [[ $ARNOLD_ID = 0 ]]; then
        #     python3 fetch_checkpoint.py --idx $1
        # else
        #     echo "wait for master to download pretrained checkpoint..."; sleep 30
        # fi
    fi
}

prepare_data(){
    local src=$1
    local tgt=$2
    local data=$3
    local data_final
    data_arrary=($(echo $data | tr ":" "\n"))
    for ele in "${data_arrary[@]}"; do
        download_if_not_exists $ele >&2
    done
    if [[ ${data_arrary[1]} != "" ]]; then
        data_final=$(combine_data $src $tgt ${data_arrary[0]} ${data_arrary[1]} ${data_arrary[2]})
    else
        data_final=${data_arrary[0]}
    fi
    echo $data_final
}


combine_data(){
    local src=$1
    local tgt=$2
    local data00=$3
    local data01=$4
    local data02=$5

    abs_data00=$(readlink -f $data00)
    if [[ ! -z $data01 ]]; then abs_data01=$(readlink -f $data01); fi
    if [[ ! -z $data02 ]]; then abs_data02=$(readlink -f $data02); fi

    echo data00 is $data00 $abs_data00 >&2
    echo data01 is $data01 $abs_data01 >&2
    echo data02 is $data02 $abs_data02 >&2

    comb_data=CMB_${data00::-4}
    if [[ ! -z $data01 ]]; then comb_data+=_AND_${data01::-4};     fi
    if [[ ! -z $data02 ]]; then comb_data+=_AND_${data02::-4};     fi

    rm -rf $comb_data
    mkdir -p $comb_data
    for lang in $src $tgt; do 
        ln -s $abs_data00/dict.$lang.txt $comb_data/dict.$lang.txt; 
        for ext in bin idx; do 
            ln -s $abs_data00/train.$src-$tgt.$lang.$ext $comb_data/train.$src-$tgt.$lang.$ext; 
            ln -s $abs_data00/valid.$src-$tgt.$lang.$ext $comb_data/valid.$src-$tgt.$lang.$ext;
            if [[ ! -z $data01 ]]; then ln -s $abs_data01/train.$src-$tgt.$lang.$ext $comb_data/train1.$src-$tgt.$lang.$ext; fi
            if [[ ! -z $data02 ]]; then ln -s $abs_data02/train.$src-$tgt.$lang.$ext $comb_data/train2.$src-$tgt.$lang.$ext; fi

            # check
            if [[ ! -z $data01 && ! -e $comb_data/train1.$src-$tgt.$lang.$ext ]]; then echo symbol link does not exists! >&2; exit 2; fi
            if [[ ! -z $data02 && ! -e $comb_data/train2.$src-$tgt.$lang.$ext ]]; then echo symbol link does not exists! >&2; exit 2; fi
        done; 
    done
    echo $comb_data
}


auto_install_dependencies(){
    if [[ "$ARNOLD_DEBUG" = "vscode" ]]
    then
        echo 'in debug mode, some steps are skipped' >&2
    else
        install_dependencies
    fi
}


install_dependencies(){
    # if [[ "$ARNOLD_DEBUG" = "vscode" ]]
    # then
    #     echo 'in debug mode, some steps are skipped' >&2
    # else
    
    pip install tensorflow
    pip install transformers
    # pip install cython
    pip install git+https://github.com/dugu9sword/lunanlp.git
    pip install git+https://github.com/dugu9sword/manytasks.git
    # pip install git+https://github.com/parlance/ctcdecode.git
    # git clone https://github.com/NVIDIA/apex
    # cd apex
    # pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
    # cd ..
    git clone https://github.com/moses-smt/mosesdecoder
    python3 setup.py build_ext --inplace
    pip install .
    # fi
}

function dist-fairseq-train(){
    if [ "$ARNOLD_DEBUG" = "vscode" ]; then
        fairseq-train "$@"
    else
        export NCCL_IB_DISABLE=0 
        export NCCL_IB_HCA=$ARNOLD_RDMA_DEVICE:1 
        export NCCL_IB_GID_INDEX=3 
        export NCCL_SOCKET_IFNAME=eth0
        python3 -m torch.distributed.launch \
            --nproc_per_node=$ARNOLD_WORKER_GPU \
            --nnodes=$ARNOLD_NUM \
            --node_rank=$ARNOLD_ID \
            --master_addr=$ARNOLD_WORKER_0_HOST \
            --master_port=$ARNOLD_WORKER_0_PORT \
            fairseq_cli/train.py \
            "$@"
    fi
}

function profile-train(){
    CUDA_VISIBLE_DEVICES=0 python3 -m cProfile -o output.pstats fairseq_cli/train.py "$@"
    gprof2dot -f pstats output.pstats | dot -Tpng -o output.png
}

function debug-train(){
    CUDA_VISIBLE_DEVICES=0 python3 -m debugpy --listen localhost:5678 fairseq_cli/train.py "$@"
}

function debug-generate(){
    CUDA_VISIBLE_DEVICES=0 python3 -m debugpy --listen localhost:5678 fairseq_cli/generate.py "$@"
}

function generate(){
    CUDA_VISIBLE_DEVICES=0 python3 fairseq_cli/generate.py "$@"
}

function parse_args(){
    while [[ "$#" -gt 0 ]]; do
        found=0
        for key in "${!BASH_ARGS[@]}"; do
            if [[ "--$key" == "$1" ]] ; then
                BASH_ARGS[$key]=$2
                found=1
            fi
        done
        if [[ $found == 0 ]]; then
            echo "arg $1 not defined!" >&2
            exit 1
        fi
        shift; shift
    done

    echo "======== PARSED BASH ARGS ========" >&2
    for key in "${!BASH_ARGS[@]}"; do
        echo "    $key = ${BASH_ARGS[$key]}" >&2
        eval "$key=${BASH_ARGS[$key]}" >&2
    done
    echo "==================================" >&2
}
