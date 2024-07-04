import os
import json
import copy
import random
import pandas as pd
from queue import Queue
from typing import List, Optional, Dict, Sequence
from dataclasses import dataclass, field

import bitsandbytes as bnb
import torch
from torch.nn.utils.rnn import pad_sequence
# from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, Dataset
import transformers


IGNORE_INDEX = -100


class KGDataset(torch.utils.data.Dataset):
    def __init__(self, examples):
        self.data = examples
        self.len = len(self.data)

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx) -> List:
        return self.data[idx]


class DataModule:
    def __init__(self, args, tokenizer: transformers.PreTrainedTokenizer, logger=None) -> None:
        self.args = args
        self.tokenizer = tokenizer

        train_examples = json.load(open(args.train_path, 'r', encoding='utf-8'))
        eval_examples = json.load(open(args.eval_path, 'r', encoding='utf-8'))
        test_examples = json.load(open(args.test_path, 'r', encoding='utf-8'))
        
        self.train_ds = KGDataset(train_examples)
        self.eval_ds = KGDataset(eval_examples)
        self.test_ds = KGDataset(test_examples)

 
@dataclass
class KGDataCollator:
    args: None
    tokenizer: transformers.PreTrainedTokenizer
    source_max_len: int
    target_max_len: int

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Extract elements
        sources = [f"{self.tokenizer.bos_token} {example['input']}" for example in instances]
        targets = [f"{example['output']} {self.tokenizer.eos_token}" for example in instances]
        # Tokenize
        tokenized_sources_with_prompt = self.tokenizer(
            sources,
            max_length=self.source_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        tokenized_targets = self.tokenizer(
            targets,
            max_length=self.target_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        source_input_ids = tokenized_sources_with_prompt['input_ids']
        target_input_ids = tokenized_targets['input_ids']

        # Build the input and labels for causal LM
        input_ids = []
        labels = []
        for tokenized_source, tokenized_target in zip(source_input_ids, target_input_ids):
            input_ids.append(
                torch.tensor(tokenized_source + tokenized_target)
            )
            labels.append(
                torch.tensor([IGNORE_INDEX for _ in range(len(tokenized_source))] + copy.deepcopy(tokenized_target))
            )

        # Apply padding
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        data_dict = {
            'input_ids': input_ids,
            'attention_mask':input_ids.ne(self.tokenizer.pad_token_id),
            'labels': labels,
        }

        # Add entity idxs to access the KGE model
        if self.args.model_class == 'KGELlama':
            data_dict['query_ids'] = torch.LongTensor([example['query_id'] for example in instances])
            data_dict['entity_ids'] = torch.LongTensor(
                [example['entity_ids'] for example in instances]
            )

        return data_dict


def make_data_module(args, tokenizer: transformers.PreTrainedTokenizer, logger=None) -> Dict:
    data_module = DataModule(args, tokenizer, logger)
    data_collator = KGDataCollator(
        args=args, tokenizer=tokenizer, 
        source_max_len=args.source_max_len, target_max_len=args.target_max_len
    )

    return {
        'train_dataset': data_module.train_ds,
        'eval_dataset': data_module.eval_ds,
        'data_collator': data_collator,
    }

