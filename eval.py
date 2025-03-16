import os
import json
import copy
import numpy as np
from time import time
from tqdm import trange, tqdm
import argparse
import pickle as pkl
from typing import Union, Dict

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import (
    HfArgumentParser, GenerationConfig, AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM, LlamaForCausalLM,
    set_seed,
)
from peft.tuners.lora import LoraLayer
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training


from arguments import ModelArguments, DataArguments, EvaluationArguments, GenerationArguments
from data import DataModule, KGDataset, KGDataCollator, IGNORE_INDEX
from utils import get_logger, print_parameter_datatypes, print_trainable_parameters
from model import EmbeddingModel, KGELlama


class Evaluator:
    def __init__(
            self, 
            args, 
            tokenizer: AutoTokenizer, 
            model: Union[AutoModelForCausalLM, PeftModel, KGELlama], 
            data_module: DataModule,
            generation_config: GenerationConfig,
    ) -> None:
        self.args = args
        self.sample_size = 200
        self.generation_config = generation_config

        self.tokenizer = tokenizer
        self.model = model
        self.data_module = data_module
        self.data_collator = KGDataCollator(args, tokenizer, args.source_max_len, args.target_max_len)


    @torch.no_grad()
    def eval_greedy(self, dataset: KGDataset):
        # self.tokenizer.padding_side = 'left'
        self.model.eval()

        preds = []
        raw_ranks = np.array([])
        ranks = np.array([])
        print_step = 1000
        data_num = len(dataset)

        for begin_idx in range(0, data_num, print_step):
            end_idx = min(begin_idx + print_step, data_num)
            generated = []
            for ex_idx, ex in enumerate(tqdm(dataset[begin_idx: end_idx])):
                prompt = ex['input']
     
                if self.args.model_class == 'LlamaForCausalLM':
                    inputs = self.tokenizer(prompt, return_tensors='pt')
                    input_ids = inputs.input_ids.cuda()  # (1, input_len)
                    input_len = input_ids.shape[-1]
                    output = self.model.generate(input_ids=input_ids, generation_config=self.generation_config)
                    generated.append(output.sequences[0, input_len:].cpu().numpy().tolist())
                if self.args.model_class == 'KGELlama':
                    inputs = self.tokenizer(prompt, return_tensors='pt')
                    input_ids = inputs.input_ids.cuda()  # (1, input_len)

                    output = self.model.generate(
                        input_ids=input_ids, 
                        query_ids=torch.LongTensor([ex['query_id']]).to(input_ids.device), 
                        entity_ids=torch.LongTensor([ex['entity_ids']]).to(input_ids.device), 
                        generation_config=self.generation_config,
                    )
                    generated.append(output.sequences[0].cpu().numpy().tolist())
                ex.pop('input')
            
            batch_preds = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
            for ex_idx, ex in enumerate(dataset[begin_idx: end_idx]):
                target = ex.pop('output')
                rank = ex['rank']
                pred = str(batch_preds[ex_idx]).strip()

                topk_names = ex['topk_names']
                if target == pred:
                    rank = 1
                else:    
                    if pred not in set(topk_names) or topk_names.index(pred) >= rank:
                        rank += 1
                
                ex['target'] = target
                ex['pred_rank'] = rank
                ex['pred'] = pred
                preds.append(ex)
                raw_ranks = np.append(raw_ranks, ex['rank'])
                ranks = np.append(ranks, rank)

            def compute_metrics(ranks_: np.ndarray):
                metrics = {
                    'hits1': np.mean(ranks_ <= 1),
                    'hits3': np.mean(ranks_ <= 3),
                    'hits10': np.mean(ranks_ <= 10),
                    'mrr': np.mean(1. / ranks_),
                }
                metrics = {k: round(v, 3) for k, v in metrics.items()}
                logger.info(f'num: {ranks_.shape[0]}; {metrics}')
            logger.info('='*80)
            compute_metrics(raw_ranks)
            compute_metrics(ranks)
        
        return preds


if __name__ == '__main__':
    set_seed(2023)

    # load args
    hfparser = HfArgumentParser((ModelArguments, DataArguments, EvaluationArguments, GenerationArguments))
    model_args, data_args, eval_args, generation_args, _ = hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
    generation_config = GenerationConfig(**vars(generation_args))
    args = argparse.Namespace(**vars(model_args), **vars(data_args), **vars(eval_args))
    assert args.model_class in ['LlamaForCausalLM', 'KGELlama']
    if args.kge_model == 'TransE':
        args.embedding_dim = 250

    # checkpoint_dir: .../checkpoint-xxxx/adapter_model
    logger = get_logger(os.path.dirname(args.checkpoint_dir))
    logger.info('args=>')
    logger.info(json.dumps(vars(args), ensure_ascii=False, indent=4))
    
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    if args.model_class == 'KGELlama':
        tokenizer.add_tokens(['[QUERY]', '[ENTITY]', '[RELATION]'])
    
    if args.model_class == 'LlamaForCausalLM':
        model = LlamaForCausalLM.from_pretrained(args.model_name_or_path, low_cpu_mem_usage=True, device_map='auto')
        model = PeftModel.from_pretrained(model, args.checkpoint_dir)
    if args.model_class == 'KGELlama':
        generation_config.bos_token_id = tokenizer.bos_token_id
        
        model = LlamaForCausalLM.from_pretrained(args.model_name_or_path, low_cpu_mem_usage=True, device_map='auto')
        model = PeftModel.from_pretrained(model, args.checkpoint_dir)
        
        llm_config = model.config
        kge_embedding_dir = os.path.join(args.dataset, args.kge_model)
        embed_model = EmbeddingModel(kge_embedding_dir, args.embedding_dim, 1024, llm_config.hidden_size, llm_config.hidden_act)
        embed_model.load_state_dict(torch.load(os.path.join(os.path.dirname(args.checkpoint_dir), 'kge.bin'), map_location='cpu'))
        
        model = KGELlama(tokenizer, model, embed_model)
    
    model.cuda()
    model.eval()

    print_parameter_datatypes(model, logger)
    
    # data
    data_module = DataModule(args, tokenizer)

    # inference
    evaluator = Evaluator(args, tokenizer, model, data_module, generation_config)
    preds = evaluator.eval_greedy(data_module.test_ds)
    output = {
        'args': vars(args),
        'generation_config': vars(generation_config),
        'prediction': preds,
    }
    output_path = os.path.join(os.path.dirname(args.checkpoint_dir), f'prediction.json')
    json.dump(output, open(output_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
