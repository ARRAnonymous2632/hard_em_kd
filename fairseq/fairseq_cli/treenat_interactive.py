#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Translate raw text with a trained model. Batches data on-the-fly.
"""

from collections import namedtuple
import fileinput

import torch

from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.data import data_utils, encoders
from fairseq.meters import StopwatchMeter


Batch = namedtuple('Batch', 'ids src_tokens src_lengths tgt_init_tokens tgt_init_lengths')
Translation = namedtuple('Translation', 'src_str hypos pos_scores alignments')


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


def make_batches(lines, cfg, task, max_positions, encode_fn):
    def encode_lines(lines):
        tokens = [
            task.source_dictionary.encode_line(
                encode_fn(src_str), add_if_not_exist=False
            ).long()
            for src_str in lines
        ]
        lengths = [t.numel() for t in tokens]
        return tokens, lengths
    batch_list = []
    for line in lines:
        dataset_id = int(line.strip())
        entry = task.datasets[cfg.gen_subset][dataset_id]
        src_tokens = entry["source"]
        tgt_tokens = entry["target"]
        src_lengths = src_tokens.shape[0]
        tgt_lengths = tgt_tokens.shape[0]
        batch_list.append(
            Batch(
                ids=[dataset_id],
                src_tokens=src_tokens.unsqueeze(0),
                src_lengths=torch.tensor(src_lengths).unsqueeze(0),
                tgt_init_tokens=tgt_tokens.unsqueeze(0),
                tgt_init_lengths=torch.tensor(tgt_lengths).unsqueeze(0)
            )
        )
    return batch_list

def build_tree(p:torch.Tensor, nodes_phrase:list, tgt_dict:dict):
    graph_lines = ["digraph {"]
    N = p.shape[0]
    p = torch.exp(p)
    for i in range(1,  N//2):
        phrase = [tgt_dict[token_idx] for token_idx in nodes_phrase[i].cpu().tolist()]
        phrase = " ".join(phrase)
        graph_lines.append(f"\t{i} [ label=\"{i} \\n r:{p[i, 0]:.3f}, l:{p[i, 1]:.3f}\\n k:{p[i, 2]:.3f}, s:{p[i, 3]:.3f} \\n \\\"{phrase}\\\"\"]")
    
    for i in range(N//2, N):
        phrase = [tgt_dict[token_idx] for token_idx in nodes_phrase[i].cpu().tolist()]
        phrase = " ".join(phrase)
        graph_lines.append(f"\t{i} [ label=\"{phrase}\"]")
    
    for i in range(N - 1, 1, -1):
        graph_lines.append(f"\t{i // 2} -> {i}")
    
    graph_lines.append("}")
    return '\n'.join(graph_lines)


def main(args):
    utils.import_user_module(args)

    if args.buffer_size < 1:
        args.buffer_size = 1
        
    max_sentences = getattr(args, 'max_sentences', None)
    max_tokens = getattr(args, 'max_tokens', None)
    if max_tokens is None and max_sentences is None:
        args.max_sentences = 1
    elif max_sentences is None:
        args.max_sentences = None

    assert not args.sampling or args.nbest == args.beam, \
        '--sampling requires --nbest to be equal to --beam'

    print(args)

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Setup task, e.g., translation
    task = tasks.setup_task(args)
    task.load_dataset(args.gen_subset)
    # Load ensemble
    print('| loading model(s) from {}'.format(args.path))
    models, _model_args = checkpoint_utils.load_model_ensemble(
        args.path.split(':'),
        arg_overrides=eval(args.model_overrides),
        task=task,
    )

    # Set dictionaries
    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary

    # Optimize ensemble for generation
    for model in models:
        model.make_generation_fast_(
            beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
            need_attn=args.print_alignment,
        )
        if args.fp16:
            model.half()
        if use_cuda:
            model.cuda()

    # Initialize generator
    generator = task.build_generator(models, args=args)

    # Handle tokenization and BPE
    tokenizer = encoders.build_tokenizer(args)
    bpe = encoders.build_bpe(args)

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
    align_dict = utils.load_align_dict(args.replace_unk)

    max_positions = utils.resolve_max_positions(
        task.max_positions(),
        *[model.max_positions() for model in models]
    )

    if args.buffer_size > 1:
        print('| Sentence buffer size:', args.buffer_size)
    print('| Type the input sentence and press return:')
    start_id = 0
    gen_timer = StopwatchMeter()
    
    for inputs in buffered_read(args.input, args.buffer_size):
        for batch in make_batches(inputs, args, task, max_positions, encode_fn):
            src_tokens = batch.src_tokens
            src_lengths = batch.src_lengths
            tgt_init_tokens = batch.tgt_init_tokens
            tgt_init_lengths = batch.tgt_init_lengths
            if use_cuda:
                src_tokens = src_tokens.cuda()
                src_lengths = src_lengths.cuda()
                tgt_init_tokens = tgt_init_tokens.cuda()
                tgt_init_lengths = tgt_init_lengths.cuda()

            sample = {
                'net_input': {
                    'src_tokens': src_tokens,
                    'src_lengths': src_lengths,
                    'tgt_init_tokens': tgt_init_tokens,
                    'tgt_init_lengths': tgt_init_lengths,
                },
            }
            gen_timer.start()
            translations, p, nodes_phrase = task.inference_step(generator, models, sample)
            num_generated_tokens = sum(len(h[0]['tokens']) for h in translations)
            gen_timer.stop(num_generated_tokens)
            results = list()
            for i, (id, hypos) in enumerate(zip(batch.ids, translations)):
                src_tokens_i = utils.strip_pad(src_tokens[i], tgt_dict.pad())
                results.append((start_id + id, src_tokens_i, p[i], nodes_phrase[i], hypos))

        # sort output to match input order
        bpe = getattr(args, 'bpe', "@@")  # "@@ "
        for id, src_tokens, p_i, nodes_phrase_i, hypos in sorted(results, key=lambda x: x[0]):
            if src_dict is not None:
                src_str = src_dict.string(src_tokens, bpe_symbol=bpe)
                print('S-{}\t{}'.format(id, src_str))

            # Process top predictions
            for hypo in hypos[:min(len(hypos), args.nbest)]:
                hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                    hypo_tokens=hypo['tokens'].int().cpu(),
                    src_str=src_str,
                    alignment=hypo['alignment'],
                    align_dict=align_dict,
                    tgt_dict=tgt_dict,
                    remove_bpe=bpe,
                )
                hypo_str = decode_fn(hypo_str)
                tree = build_tree(p_i, nodes_phrase_i, tgt_dict)

                print('H-{}\t{}\t{}'.format(id, hypo['score'], hypo_str))
                print('P-{}\t{}'.format(
                    id,
                    ' '.join(map(lambda x: '{:.4f}'.format(x), hypo['positional_scores'].tolist()))
                ))
                if args.print_alignment:
                    alignment_str = " ".join(["{}-{}".format(src, tgt) for src, tgt in alignment])
                    print('A-{}\t{}'.format(
                        id,
                        alignment_str
                    ))
                print(f"G-{id} \n {tree}\n\n")
        # update running id counter
        start_id += len(inputs)
    print('| Translated {} sentences ({} tokens) in {:.1f}s ({:.2f} sentences/s, {:.2f} tokens/s)'.format(
        start_id, gen_timer.n, gen_timer.sum, start_id / gen_timer.sum, 1. / gen_timer.avg))


def cli_main():
    parser = options.get_generation_parser(interactive=True)
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == '__main__':
    cli_main()