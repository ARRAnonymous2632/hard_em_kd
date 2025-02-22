CUDA_VISIBLE_DEVICES=0,1,2,3 python qwen_batch_inference.py \
    --prompt prompts/ro-en_prompt.txt \
    --hypos ./data/raw_ro_en/kd.out.shard.out.H.2 \
    --sources ./data/raw_ro_en/kd.out.shard.out.S.2 \
    --targets ./data/raw_ro_en/kd.out.shard.out.T.2 \
    --use_targets 0 --output_dir ./results/ro_en_fixed_qwen/0 --num_gpus 4