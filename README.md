Implementation of "Alleviating the Multimodality Issue in Autoregressive Machine Translation: A Hard-EM Learning Framework"

In this repo, 
1. `fairseq` contains the script for Transformer-based sequence-to-sequence training.
2. `llm_correction` shows how we prompt QWen 2.5 to correct the translations.
3. `data` has the following data files: 
  - `hard_em_itr1.en` `hard_em_itr1.de`
  - `hard_em_itr2.en` `hard_em_itr2.de`
  - `bpe.code` 

# Instruction
## Fairseq Training 
1. Follow `fairseq/new_setup.py` for installing the environment.
2. Follow the official [fairseq instruction](https://github.com/facebookresearch/fairseq/tree/main/examples/translation#wmt14-english-to-german-convolutional) for preprocessing.
3. Train with our configurations (under `fairseq/config/hard_em`).
For example, to train the baseline model, under the `fairseq` directory, run:
``` 
bash trainer_with_config.sh config/config_llm_at/wmt14ende_raw_base.yml
```
> Note: `data_bin` in the configurations files needs to be changed accordingly
4. For evaluation:
```
bash eval_at.sh [your_checkpoint_path]
```
To gather model's generation over the training sets, an example script can be found at: `fairseq/datasets/wmt14_en_de_kd/s5_kd.sh`

## LLM Inference
Given parall files of source.txt, hypothesis.txt, and target.txt, 
Under `llm_correction` run 
```bash 
bash entry_point_qwen.sh \
prompt=$1  \
hypos=$2 \
targets=$3 \
sources=$4 \
EXPERIMENT_DIR=$5
```
where `prompt` is given in `llm_correction/prompts`, and is explained in the appendix of our paper.
