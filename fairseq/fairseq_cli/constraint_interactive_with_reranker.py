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
import time

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


def make_batches(lines, args, task, max_positions, encode_fn, init_outputs_list=[]):
    tokens = [
        task.source_dictionary.encode_line_with_bos(
            encode_fn(src_str), add_if_not_exist=False, append_bos=True,
        ).long()
        for src_str in lines
    ]
    init_outputs = []
    init_lengths = []
    for init_output_strs in init_outputs_list:
        line_init_outputs = " ".join(init_output_strs)
        line_init_lengths = torch.LongTensor(
            [len(s.split()) for s in init_output_strs]
        )
        init_outputs.append(task.target_dictionary.encode_line_with_bos(
                                encode_fn(line_init_outputs), add_if_not_exist=False, append_bos=False, append_eos=False
                           ).long())
        init_lengths.append(line_init_lengths)
    lengths = torch.LongTensor([t.numel() for t in tokens])
    itr = task.get_batch_iterator(
        dataset=task.build_dataset_for_inference(tokens, lengths),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=max_positions,
    ).next_epoch_itr(shuffle=False)
    for batch in itr:
        yield Batch(
            ids=batch['id'],
            src_tokens=batch['net_input']['src_tokens'], src_lengths=batch['net_input']['src_lengths'],
            tgt_init_tokens=data_utils.collate_tokens(
                [init_outputs[idx] for idx in batch['id']],
                task.target_dictionary.pad(), task.target_dictionary.eos()
            ),
            tgt_init_lengths=data_utils.collate_tokens(
                [init_lengths[idx] for idx in batch['id']], 0
            )
        )


def main(args):
    utils.import_user_module(args)

    if args.buffer_size < 1:
        args.buffer_size = 1
        
    max_sentences = getattr(args, 'max_sentences', None)
    max_tokens = getattr(args, 'max_tokens', None)
    if max_tokens is None and max_sentences is None:
        args.max_sentences = 1

    assert not args.sampling or args.nbest == args.beam, \
        '--sampling requires --nbest to be equal to --beam'
    assert not args.max_sentences or args.max_sentences <= args.buffer_size, \
        '--max-sentences/--batch-size cannot be larger than --buffer-size'

    print(args)

    use_cuda = torch.cuda.is_available() and not args.cpu

      # Setup task, e.g., translation
    args.task = 'translation_lex_control' # override task
    task_lex_control = tasks.setup_task(args)
    args.task = 'translation_rouge' # override task
    task_translation = tasks.setup_task(args)

    # Setup task, e.g., translation
    # task_translation = tasks.setup_task(args)

    # Load ensemble
    print('| loading model(s) from {}'.format(args.path))
    model_lexical_control, _model_args = checkpoint_utils.load_model_ensemble(
        [args.path.split(':')[0]],
        arg_overrides=eval(args.model_overrides),
        task=task_lex_control,
    )
    
    args.arch = 'lstm_small'
    model_reranker, _model_args = checkpoint_utils.load_model_ensemble(
        [args.path.split(':')[1]],
        # arg_overrides=eval(args.model_overrides),
        task=task_translation,
    )
    
    models = [model_lexical_control[0], model_reranker[0]] # 
    # Set dictionaries
    src_dict = task_lex_control.source_dictionary
    tgt_dict = task_lex_control.target_dictionary

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

    models[0].to('cuda:0')
    models[1].to('cuda:1')
    
    # Initialize generator
    generator = task_lex_control.build_generator(args=args)

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
        task_lex_control.max_positions(),
        *[model.max_positions() for model in models]
    )

    if args.buffer_size > 1:
        print('| Sentence buffer size:', args.buffer_size)
    print('| Type the input sentence and press return:')
    start_id = 0
    gen_timer = StopwatchMeter()
    
    def get_id_from_restore_file(restore_file_path):
        with open(restore_file_path, 'r') as f:
            lines = f.readlines()
            id_list = []
            for line in lines:
                if line.startswith('S-'):
                    id_list.append(int(line.split('\t')[0].split('-')[1]))
            return id_list
    if False:
        restore_file_path = 'z_output_lexical_shift/newstest2017-dinu-iate.de._bs_10_4_consecutive_rerank_1.txt'
        id_list = get_id_from_restore_file(restore_file_path)
    else:
        id_list = None
    
    for inputs in buffered_read(args.input, args.buffer_size):
        results = []

        # input is sentence \t s1|||t1 \t s2|||t2 ...
        new_inputs = []
        constraints = []
        
        for inp in inputs:
            inp = inp.split('\t')
            new_inputs.append(inp[0])
            constraints.append([tup.split('|||')[1] for tup in inp[1:]])

        for batch in make_batches(new_inputs, args, task_lex_control, max_positions, encode_fn, constraints):
            start_time = time.time()  # Record the start time for this iteration

            id = batch.ids[0].int()
            if id_list is not None:
                if id in id_list:
                    print('Skipping id:', id)
                    continue
            
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
            
            translations = task_lex_control.inference_step_with_reranker(generator, models, sample)
            num_generated_tokens = sum(len(h['tokens']) for h in translations)
            gen_timer.stop(num_generated_tokens)
            
            
            src_str = src_dict.string(src_tokens, bpe_symbol=bpe)
            print('S-{}\t{}'.format(id, src_str))

            for i, hypo in  enumerate(translations):                
                hypo_tokens = hypo['tokens'].int().cpu()
                hypo_tokens = hypo_tokens[hypo_tokens!=tgt_dict.pad()]
                hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                    hypo_tokens=hypo_tokens,
                    src_str=src_str,
                    alignment=None,
                    align_dict=None,
                    tgt_dict=tgt_dict,
                    remove_bpe=bpe,
                )
                hypo_str = decode_fn(hypo_str)
                print('H-{}-{}\t{}\t{}'.format(id, i, hypo['score'], hypo_str))
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Time elapsed for iteration {i+1}: {elapsed_time:.4f} seconds")

        start_id += len(inputs)
    print('| Translated {} sentences ({} tokens) in {:.1f}s ({:.2f} sentences/s, {:.2f} tokens/s)'.format(
        start_id, gen_timer.n, gen_timer.sum, start_id / gen_timer.sum, 1. / gen_timer.avg))


def cli_main():
    parser = options.get_generation_parser(interactive=True)
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == '__main__':
    cli_main()