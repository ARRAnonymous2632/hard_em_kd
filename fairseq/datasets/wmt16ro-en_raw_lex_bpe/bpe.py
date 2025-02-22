bpe_codes_path = 'data-bin/wmt16ro-en_raw_lex_bpe/bpe.codes.roen'  # Path to your BPE codes file


import codecs
from subword_nmt.apply_bpe import BPE
from sacremoses import MosesTokenizer
from tqdm import tqdm 
import os

def write_lines(lines, path):
    with open(path, "w", encoding="utf-8") as f:
        lines = "\n".join(lines)
        f.writelines(lines)

def read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        data = f.readlines()
        data = [line.strip() for line in data]
    return data

folder = "wmt16ro-en_raw_lex_bpe"
src_hypo_pairs = [ 
    ("valid", f"datasets/{folder}/data/valid.ro-en.ro.txt", f"datasets/{folder}/data/valid.ro-en.en.txt"), 
    ("test", f"datasets/{folder}/data/test.ro-en.ro.txt", f"datasets/{folder}/data/test.ro-en.en.txt"),
    ("train", f"datasets/{folder}/data/train.ro-en.ro.txt", f"datasets/{folder}/data/train.ro-en.en.txt")
    ]

src_lang="ro"
tgt_lang="en"
lang_pair="ro-en"
output_folder=f"datasets/{folder}/data"


# Load BPE codes
with codecs.open(bpe_codes_path, encoding='utf-8') as codes_file:
    bpe = BPE(codes_file, separator='@@')

# Load vocabulary
moses_tokenizer_ro = MosesTokenizer('ro')
moses_tokenizer_en = MosesTokenizer('en')


for split, src_path, hypo_path in src_hypo_pairs:
    src = read_file(src_path)
    hypo = read_file(hypo_path)
    size = len(src)
    for i in tqdm(range(size)):
        src_tokens = bpe.process_line(' '.join(moses_tokenizer_ro.tokenize(src[i].strip(), aggressive_dash_splits=True)))
        hypo_tokens = bpe.process_line(' '.join(moses_tokenizer_en.tokenize(hypo[i].strip(), aggressive_dash_splits=True)))

        src[i] = src_tokens
        hypo[i] = hypo_tokens


    write_lines(src, os.path.join(output_folder, f"{split}.{lang_pair}.tok.{src_lang}"))
    write_lines(hypo, os.path.join(output_folder, f"{split}.{lang_pair}.tok.{tgt_lang}"))