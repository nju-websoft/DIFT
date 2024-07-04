from genericpath import isfile
import os
import json
import copy
import tqdm
import logging
import pandas as pd
import argparse
from typing import List, Optional, Dict, Sequence, Union
from dataclasses import dataclass, field

import bitsandbytes as bnb
import torch

import transformers
from transformers import (
    AutoConfig,  GenerationConfig,
    AutoTokenizer, PreTrainedTokenizer, 
    AutoModelForCausalLM,
    Seq2SeqTrainingArguments, Seq2SeqTrainer, HfArgumentParser,
    set_seed,
    Seq2SeqTrainer,
    BitsAndBytesConfig,
    LlamaTokenizer
)

from peft.tuners.lora import LoraLayer
from peft import LoraConfig, get_peft_model, PeftModelForCausalLM, prepare_model_for_kbit_training
from model import KGELlama


def find_all_linear_names(args, model):
    if args.use_quant:
        cls = bnb.nn.Linear4bit if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    else:
        cls = torch.nn.Linear

    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def get_accelerate_model(args, config, pretrained_model_class):
    # if we are in a distributed setting, we need to set the device map and max memory per device
    device_map = 'auto' if os.environ.get('LOCAL_RANK') is None else {'': int(os.environ.get('LOCAL_RANK', '0'))}

    print(f'Loading base model {args.model_name_or_path}...')
    if args.use_quant:
        compute_dtype = (torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))
        model = pretrained_model_class.from_pretrained(
            args.model_name_or_path,
            config=config,
            load_in_4bit=args.bits == 4,
            load_in_8bit=args.bits == 8,
            device_map='auto',
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=args.bits == 4,
                load_in_8bit=args.bits == 8,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=args.double_quant,
                bnb_4bit_quant_type=args.quant_type,
            ),
            torch_dtype=(torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)),
        )
    else:
        model = pretrained_model_class.from_pretrained(
            args.model_name_or_path, 
            config=config,
            low_cpu_mem_usage=True, 
            device_map=device_map, 
        )

    setattr(model, 'model_parallel', True)
    setattr(model, 'is_parallelizable', True)

    if not args.full_finetune:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.use_quant)
        print(f'Adding LoRA modules...')
        config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)

    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if args.bf16:
                module = module.to(torch.bfloat16)
        if 'norm' in name:
            module = module.to(torch.float32)
        if 'lm_head' in name or 'embed_tokens' in name:
            if hasattr(module, 'weight'):
                if args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)
    return model


class SavePeftModelCallback(transformers.TrainerCallback):
    def save_model(self, args, state, kwargs):
        print('Saving PEFT checkpoint...')
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(state.best_model_checkpoint, "adapter_model")
        else:
            checkpoint_folder = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        for file_name in os.listdir(checkpoint_folder):
            if 'kge' in file_name:
                continue
            file_path = os.path.join(checkpoint_folder, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)

    def on_train_end(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)


def print_trainable_parameters(args, model, logger=None):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    if args.use_quant and args.bits == 4: trainable_params /= 2

    trainable = round(100 * trainable_params / all_params, 3)
    trainable_params = trainable_params//10**6
    all_params = all_params//10**9
    
    if logger is None:
        print(f"trainable params: {trainable_params}MB || all params: {all_params}GB || trainable: {trainable}%")
    else:
        logger.info(f"trainable params: {trainable_params}MB || all params: {all_params}GB || trainable: {trainable}%")

def print_parameter_datatypes(model, logger=None):
    dtypes = dict()
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes: dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    
    total = 0
    for k, v in dtypes.items(): total += v

    for k, v in dtypes.items():

        if logger is None:
            print(f'type: {k} || num: {v} || {round(v/total, 3)}')
        else:
            logger.info(f'type: {k} || num: {v} || {round(v/total, 3)}')


def get_logger(log_dir: str):
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    log_file = os.path.join(log_dir, 'log.txt')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(log_format)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_format)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger