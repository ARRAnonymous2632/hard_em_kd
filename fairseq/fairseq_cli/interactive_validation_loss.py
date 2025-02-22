#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Translate raw text with a trained model. Batches data on-the-fly.
"""

import ast
import fileinput
import logging
import math
import os
import sys
import time
from argparse import Namespace
from collections import namedtuple

import numpy as np
import torch

from fairseq import checkpoint_utils, distributed_utils, options, tasks, utils
from fairseq.dataclass.configs import FairseqConfig
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.token_generation_constraints import pack_constraints, unpack_constraints
from fairseq_cli.generate import get_symbols_to_strip_from_output
from fairseq.data.language_pair_dataset import LanguagePairDataset
from fairseq.logging import progress_bar
from fairseq.data import data_utils

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.interactive")


Batch = namedtuple("Batch", "ids src_tokens src_lengths constraints")
Translation = namedtuple("Translation", "src_str hypos pos_scores alignments")


def buffered_read(input, buffer_size):
    buffer = []
    with fileinput.input(files=[input], openhook=fileinput.hook_encoded("utf-8")) as h:
        for src_str in h:
            buffer.append(src_str.strip())
            if len(buffer) >= buffer_size:
                yield buffer
                buffer = []

    if len(buffer) > 0:
        yield buffer

import pickle

def make_batch_iterator(cfg, task, max_positions, encode_fn):
    if cfg.interactive.input.endswith("pkl"):
        with open(cfg.interactive.input, "rb") as f:
            data = pickle.load(f)
        src_tokens = data["src_tokens"]
        src_lengths = data["src_lengths"]
        tgt_tokens = data["tgt_tokens"]
        tgt_lengths = data["tgt_lengths"]
    elif os.path.isdir(cfg.interactive.input):
        #TODO load the binary files  
        lang_pair = f"{task.cfg.source_lang}-{task.cfg.target_lang}"
        
        suffix = f"{cfg.dataset.gen_subset}.{lang_pair}.{task.cfg.source_lang}"
        src_path = os.path.join(cfg.interactive.input, suffix)
        src_dataset = data_utils.load_indexed_dataset(src_path, task.source_dictionary, task.cfg.dataset_impl)
        suffix = f"{cfg.dataset.gen_subset}.{lang_pair}.{task.cfg.target_lang}"
        tgt_path = os.path.join(cfg.interactive.input, suffix)
        tgt_dataset = data_utils.load_indexed_dataset(tgt_path, task.target_dictionary, task.cfg.dataset_impl)
        pair_dset = LanguagePairDataset(
            src_dataset,
            src_dataset.sizes,
            task.source_dictionary,
            tgt_dataset,
            tgt_dataset.sizes,
            task.target_dictionary)

        return task.get_batch_iterator(
                    dataset=pair_dset,
                    max_tokens=cfg.dataset.max_tokens,
                    max_sentences=cfg.dataset.batch_size,
                    max_positions=max_positions,
                    ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
                ).next_epoch_itr(shuffle=False) 
    else:
        with open(cfg.interactive.input, "r", encoding="utf-8") as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
        src = [line.split("|||")[0] for line in lines]
        tgt = [line.split("|||")[1] for line in lines]
        src_tokens, src_lengths = task.get_interactive_tokens_and_lengths(src, encode_fn)
        tgt_tokens, tgt_lengths = task.get_interactive_tokens_and_lengths(tgt, encode_fn)
        pickle_path = os.path.dirname(cfg.interactive.input)
        pickle_path = os.path.join(pickle_path, f"{os.path.basename(cfg.interactive.input)}.pkl")
        data = {"src_tokens": src_tokens, "tgt_tokens": tgt_tokens, "src_lengths": src_lengths, "tgt_lengths": tgt_lengths}
        with open(pickle_path, "wb") as f:
            pickle.dump(data, f)
        
    itr = task.get_batch_iterator(
        dataset=LanguagePairDataset(
                        src=src_tokens, 
                        src_sizes=src_lengths, 
                        tgt=tgt_tokens, 
                        tgt_sizes=tgt_lengths,
                        src_dict=task.source_dictionary,
                        tgt_dict=task.target_dictionary
                    ),
        max_tokens=cfg.dataset.max_tokens,
        max_sentences=cfg.dataset.batch_size,
        max_positions=max_positions,
        ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
    ).next_epoch_itr(shuffle=False)
    return itr

def main(cfg: FairseqConfig):
    if isinstance(cfg, Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    start_time = time.time()
    total_translate_time = 0

    utils.import_user_module(cfg.common)

    if cfg.interactive.buffer_size < 1:
        cfg.interactive.buffer_size = 1
    if cfg.dataset.max_tokens is None and cfg.dataset.batch_size is None:
        cfg.dataset.batch_size = 1

    assert (
        not cfg.generation.sampling or cfg.generation.nbest == cfg.generation.beam
    ), "--sampling requires --nbest to be equal to --beam"
    assert (
        not cfg.dataset.batch_size
        or cfg.dataset.batch_size <= cfg.interactive.buffer_size
    ), "--batch-size cannot be larger than --buffer-size"

    logger.info(cfg)

    # Fix seed for stochastic decoding
    if cfg.common.seed is not None and not cfg.generation.no_seed_provided:
        np.random.seed(cfg.common.seed)
        utils.set_torch_seed(cfg.common.seed)

    use_cuda = torch.cuda.is_available() and not cfg.common.cpu

    # Setup task, e.g., translation
    task = tasks.setup_task(cfg.task)

    # Load ensemble
    overrides = ast.literal_eval(cfg.common_eval.model_overrides)
    logger.info("loading model(s) from {}".format(cfg.common_eval.path))
    models, _model_args = checkpoint_utils.load_model_ensemble(
        utils.split_paths(cfg.common_eval.path),
        arg_overrides=overrides,
        task=task,
        suffix=cfg.checkpoint.checkpoint_suffix,
        strict=(cfg.checkpoint.checkpoint_shard_count == 1),
        num_shards=cfg.checkpoint.checkpoint_shard_count,
    )

    # Set dictionaries
    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary

    # Optimize ensemble for generation
    for model in models:
        if model is None:
            continue
        if cfg.common.fp16:
            model.half()
        if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
            model.cuda()
        model.prepare_for_inference_(cfg)

    # Initialize generator
    generator = task.build_generator(models, cfg.generation)

    # Handle tokenization and BPE
    tokenizer = task.build_tokenizer(cfg.tokenizer)
    bpe = task.build_bpe(cfg.bpe)

    def encode_fn(x):
        if tokenizer is not None:
            x = tokenizer.encode(x)
        if bpe is not None:
            x = bpe.encode(x)
        return x

    def decode_fn(x):
        if bpe is not None:
            x = bpe.decode(x)
        if tokenizer is not None:
            x = tokenizer.decode(x)
        return x

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(cfg.generation.replace_unk)

    max_positions = utils.resolve_max_positions(
        task.max_positions(), *[model.max_positions() for model in models]
    )

    if cfg.generation.constraints:
        logger.warning(
            "NOTE: Constrained decoding currently assumes a shared subword vocabulary."
        )
    
    
    itr = make_batch_iterator(cfg, task, max_positions, encode_fn)
    progress = progress_bar.progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_interval=cfg.common.log_interval,
        default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
    )
    criterion = task.build_criterion(cfg.criterion)
    for sample in progress:
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        loss_values = criterion.get_per_sample_loss(model, sample)
        for idx, id in enumerate(sample["id"].cpu().tolist()):
            print(f"LOSS-{id}\t{loss_values[idx].item()}")
            # entropy_str = f"ENT-{id}"
            # for i in range(entropy.shape[1]):
            #     if entropy[idx, i].item() == -10.0:
            #         break
            #     entropy_str += f"\t{entropy[idx, i].item():.5f}"
            # print(entropy_str)



def cli_main():
    parser = options.get_interactive_generation_parser()
    args = options.parse_args_and_arch(parser)
    distributed_utils.call_main(convert_namespace_to_omegaconf(args), main)


if __name__ == "__main__":
    cli_main()
