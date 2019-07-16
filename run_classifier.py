# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import sys
import random

import numpy as np

import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from torch.nn import CrossEntropyLoss

from modeling import BertForTaskNLU
from tokenization import BertTokenizer
from optimization import BertAdam, WarmupLinearSchedule

from run_classifier_dataset_utils import processors, convert_examples_to_features, write_result

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

WEIGHTS_NAME = 'pytorch_model.bin'

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data",
                        default=None,
                        type=str,
                        help="Train data path.")
    parser.add_argument("--test_data",
                        default=None,
                        type=str,
                        help="Test data path.")
    parser.add_argument("--eval_data",
                        default=None,
                        type=str,
                        help="Eval data path.")
    parser.add_argument("--bert_model", 
                        default=None, 
                        type=str, 
                        required=True,
                        help="PreTrained model path.")
    parser.add_argument("--config", 
                        default=None, 
                        type=str, 
                        required=True,
                        help="Model config path.")
    parser.add_argument("--vocab", 
                        default=None, 
                        type=str, 
                        required=True,
                        help="Vocabulary path.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--result_file",
                        default=None,
                        type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--dic_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The dic directory which used by rule.")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_predict",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--pred_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--overwrite_output_dir',
                        action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=2019,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    args = parser.parse_args()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    args.device = device

    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_predict:
        raise ValueError("At least one of `do_train` or `do_predict` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()

    label_list = processor.get_labels()
    domain_map = {}
    for (i, label) in enumerate(label_list['domain']):
        domain_map[label] = i

    intent_map = {}
    for (i, label) in enumerate(label_list['intent']):
        intent_map[label] = i

    slots_map = {}
    for (i, label) in enumerate(label_list['slots']):
        slots_map[label] = i

    logger.info("***** label list *****")
    for key, value in label_list.items():
        logger.info("%s(%d): %s" %(key, len(value), ", ".join(value)))

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    tokenizer = BertTokenizer.from_pretrained(args.vocab, do_lower_case=args.do_lower_case)
    model = BertForTaskNLU.from_pretrained(args.bert_model, args.config, label_list=label_list, max_seq_len=args.max_seq_length)
    if args.local_rank == 0:
        torch.distributed.barrier()

    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0

    if args.do_train:
        # Prepare data loader
        train_examples = processor.get_train_examples(args.train_data)
        random.seed(args.seed)
        random.shuffle(train_examples)
        train_features = convert_examples_to_features(
            train_examples, domain_map, intent_map, slots_map, args.max_seq_length, tokenizer)
            
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_domain_ids = torch.tensor([f.domain_id for f in train_features], dtype=torch.long)
        all_intent_ids = torch.tensor([f.intent_id for f in train_features], dtype=torch.long)
        all_slots_ids = torch.tensor([f.slots_id for f in train_features], dtype=torch.long)

        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_domain_ids, all_intent_ids, all_slots_ids)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        num_train_optimization_steps = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

        # Prepare optimizer

        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        if args.fp16:
            try:
                from apex.optimizers import FP16_Optimizer
                from apex.optimizers import FusedAdam
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

            optimizer = FusedAdam(optimizer_grouped_parameters,
                                  lr=args.learning_rate,
                                  bias_correction=False,
                                  max_grad_norm=1.0)
            if args.loss_scale == 0:
                optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
            else:
                optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
            warmup_linear = WarmupLinearSchedule(warmup=args.warmup_proportion,
                                                 t_total=num_train_optimization_steps)

        else:
            optimizer = BertAdam(optimizer_grouped_parameters,
                                 lr=args.learning_rate,
                                 warmup=args.warmup_proportion,
                                 t_total=num_train_optimization_steps)

        logger.info("***** Running training *****")
        logger.info("Num examples = %d", len(train_examples))
        logger.info("Batch size = %d", args.train_batch_size)
        logger.info("Num steps = %d", num_train_optimization_steps)

        model.train()
        for _ in range(int(args.num_train_epochs)):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, domain_id, intent_id, slots_id = batch

                # define a new function to compute loss values for both output_modes
                domain_logits, intent_logits, slots_logits = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask)

                loss_fct = CrossEntropyLoss()
                loss = loss_fct(domain_logits, domain_id) + loss_fct(intent_logits, intent_id)
                for i in range(len(slots_id)):
                    loss += loss_fct(slots_logits[i], slots_id[i])
                    
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used that handles this automatically
                        lr_this_step = args.learning_rate * warmup_linear.get_lr(global_step, args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
                    if args.local_rank in [-1, 0] and nb_tr_steps % 20 == 0:
                        # logger.info("lr = {}, global_step = {}".format(optimizer.get_lr()[0], global_step))
                        logger.info("loss = {:.6f}, global_step = {}".format(tr_loss/global_step, global_step))

    ### Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    ### Example:
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Save a trained model, configuration and tokenizer
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)      
    else:
        model = BertForTaskNLU.from_pretrained(args.bert_model, args.config, label_list=label_list, max_seq_len=args.max_seq_length)

    model.to(device)

    ### prediction
    if args.do_predict and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        pred_examples = processor.get_test_examples(args.test_data)
        pred_features = convert_examples_to_features(
            pred_examples, domain_map, intent_map, slots_map, args.max_seq_length, tokenizer)
        
        logger.info("***** Running prediction *****")
        logger.info("Num examples = %d", len(pred_examples))
        logger.info("Batch size = %d", args.pred_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in pred_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in pred_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in pred_features], dtype=torch.long)

        pred_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
        # Run prediction for full data
        if args.local_rank == -1:
            pred_sampler = SequentialSampler(pred_data)
        else:
            pred_sampler = DistributedSampler(pred_data)  # Note that this sampler samples randomly
        pred_dataloader = DataLoader(pred_data, sampler=pred_sampler, batch_size=args.pred_batch_size)

        model.eval()
        preds = []

        for input_ids, input_mask, segment_ids in pred_dataloader:
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)

            with torch.no_grad():
                domain_logits, intent_logits, slots_logits = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
            domain = domain_logits.detach().cpu().numpy()
            intent = intent_logits.detach().cpu().numpy()
            slots = slots_logits.detach().cpu().numpy()
            for i in range(domain.shape[0]):
                preds.append({"domain":domain[i], "intent":intent[i], "slots":slots[i]})

        output_predict_file = os.path.join(args.output_dir, args.result_file)
        write_result(output_predict_file, args.dic_dir, preds, pred_examples, domain_map, intent_map, slots_map)

if __name__ == "__main__":
    main()
