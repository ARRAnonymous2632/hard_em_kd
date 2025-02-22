from langchain_community.llms.vllm import VLLM
from langchain_core.prompts import PromptTemplate
import transformers
import random
import json
import argparse
import os
transformers.utils.logging.set_verbosity_error()
from tqdm import tqdm
import logging
MODEL_ID = "Qwen/Qwen2.5-32B-Instruct"

from dataset import Dataset

def build_llm(args):
    return VLLM( model=MODEL_ID,
                trust_remote_code=True,  # mandatory for hf models
                tensor_parallel_size=args.num_gpus,
                max_new_tokens=args.max_new_tokens,
                top_k=args.top_k,
                top_p=args.top_p,
                temperature=args.temperature,
                vllm_kwargs={"disable_custom_all_reduce": True, "seed": args.seed}
            )

def format_response(responses, ids):
    result = []
    for idx, response in enumerate(responses):
        result.append({"ids": ids[idx], "response": response})
    return result

def main(args):

    dataset = Dataset(args.sources, args.hypos, args.targets, use_ground_truth=(args.use_targets>1), direct_translation=(args.dt > 0))
    llm = build_llm(args)
    template = open(args.prompt, "r").read()

    prompt = PromptTemplate.from_template(template)
    indices = [i for i in range(dataset.get_batch_count())]
    if args.samples > -1:
        indices = random.sample(indices, args.samples)
    else:
        indices = indices

    if args.end_idx == -1:
        args.end_idx = len(indices)

    if args.start_idx != -1:
        indices = indices[args.start_idx:args.end_idx]

    BATCH_SIZE = 128
    for s in range(0, len(indices), BATCH_SIZE):
        end = s + BATCH_SIZE
        
        batch = [
            dataset.get_batch(i) for i in indices[s:end]
        ]

        ids = [b[1] for b in batch]
        prompts = [prompt.format_prompt(samples="\n".join(b[0])) for b in batch]

        responses = llm.batch(prompts, config={"use_tqdm":False})
        results = format_response(responses, ids)
        if s % (100 * BATCH_SIZE) == 0:
            logging.info(f"processsed batch: {s//BATCH_SIZE}")
        
        with open(os.path.join(args.output_dir, f"{str(s//BATCH_SIZE)}.json"), "w", encoding="utf-8") as f:
            json.dump(results, f)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--seed", type=int, default=6547)
    argparser.add_argument("--samples", type=int, default=-1)
    argparser.add_argument("--start_idx", type=int,  default=-1)
    argparser.add_argument("--end_idx", type=int, default=-1)
    argparser.add_argument("--max_new_tokens", type=int, default=2048)
    argparser.add_argument("--top_k", type=int, default=10)
    argparser.add_argument("--top_p", type=int, default=0.95)
    argparser.add_argument("--temperature", type=int, default=0.8)
    argparser.add_argument("--num_gpus", type=int, default=1)

    argparser.add_argument("--prompt", required=True)
    argparser.add_argument("--hypos", required=True)
    argparser.add_argument("--targets", required=True)
    argparser.add_argument("--output_dir", required=True)
    argparser.add_argument("--sources", default=None)
    argparser.add_argument("--use_targets", type=int, default=0)
    argparser.add_argument("--dt", type=int, default=0)

    args = argparser.parse_args()
    random.seed(args.seed)

    main(args)