import os
import json
import copy
import tqdm
import pandas as pd
import argparse
from typing import List, Optional, Dict, Sequence
from dataclasses import dataclass, field

import bitsandbytes as bnb
import torch
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset, Dataset

import transformers
from transformers import AutoConfig,  GenerationConfig
from transformers import AutoTokenizer, PreTrainedTokenizer
from transformers import AutoModelForCausalLM, LlamaForCausalLM
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, HfArgumentParser
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers import (
    set_seed,
    Seq2SeqTrainer,
)

from arguments import ModelArguments, DataArguments, TrainingArguments, GenerationArguments
from data import DataModule, make_data_module
from model import EmbeddingModel, KGELlama
from utils import SavePeftModelCallback, print_trainable_parameters, print_parameter_datatypes, get_logger, get_accelerate_model


def train():
    hfparser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, GenerationArguments))
    model_args, data_args, training_args, generation_args, _ = hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
    training_args.generation_config = GenerationConfig(**vars(generation_args))
    args = argparse.Namespace(**vars(model_args), **vars(data_args), **vars(training_args))
    assert args.model_class in ['LlamaForCausalLM', 'KGELlama']
    if args.kge_model == 'TransE':
        args.embedding_dim = 250
    
    set_seed(args.seed)
    os.makedirs(args.output_dir)
    logger = get_logger(args.output_dir)
    logger.info(vars(args))
    
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    if args.model_class == 'KGELlama':
        tokenizer.add_tokens(['[QUERY]', '[ENTITY]', '[RELATION]'])

    model_config = AutoConfig.from_pretrained(args.model_name_or_path)
    model = get_accelerate_model(args, model_config, LlamaForCausalLM)
    model.config.use_cache = False

    if args.model_class == 'KGELlama':
        llm_config = model.config
        kge_embedding_dir = os.path.join(args.dataset, args.kge_model)
        embed_model = EmbeddingModel(kge_embedding_dir, args.embedding_dim, 1024, llm_config.hidden_size, llm_config.hidden_act)
        model = KGELlama(tokenizer, model, embed_model)

    # Verifying the datatypes and parameter counts before training.
    print_trainable_parameters(args, model, logger)
    print_parameter_datatypes(model, logger)

    data_module = make_data_module(args, tokenizer, logger)
    
    trainer = Seq2SeqTrainer(
        model=model, 
        tokenizer=tokenizer, 
        args=training_args, 
        **data_module,
    )
    if not args.full_finetune:
        trainer.add_callback(SavePeftModelCallback)
    
    # Training
    if args.do_train:
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state() 

if __name__ == '__main__':
    train()

