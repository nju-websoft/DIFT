import os
import json
from typing import List, Optional, Dict, Sequence
from dataclasses import dataclass, field
from transformers import Seq2SeqTrainingArguments


@dataclass
class ModelArguments:
    model_class: str = field(
        default='KGELlama',
        metadata={"help": "LlamaForCausalLM | KGELlama"}
    )
    model_name_or_path: Optional[str] = field(
        default="llama-2-7b-chat-hf",
        metadata={"help": "The directory in which LLM saved"}
    )
    kge_model: Optional[str] = field(
        default="CoLE",
        metadata={"help": "which pretrained embeddings to use"}
    )
    embedding_dim: int = field(default=768, metadata={'help': 'embedding dim for kge model'})

@dataclass
class DataArguments:
    dataset: str = field(default=None, metadata={"help": "Which dataset to finetune on."})
    train_path: str = field(default=None, metadata={"help": "path for train file."})
    eval_path: str = field(default=None, metadata={"help": "path for valid file."})
    test_path: str = field(default=None, metadata={"help": "path for test file."})
     
    source_max_len: int = field(default=2048, metadata={"help": "Maximum source sequence length."},)
    target_max_len: int = field(default=64, metadata={"help": "Maximum target sequence length."},)

@dataclass
class TrainingArguments(Seq2SeqTrainingArguments):
    full_finetune: bool = field(default=False, metadata={"help": "Finetune the entire model without adapters."})
    use_quant: bool = field(default=False, metadata={"help": "Finetune the entire model without adapters."})
    double_quant: bool = field(default=True, metadata={"help": "Compress the quantization statistics through double quantization."})
    quant_type: str = field(default="nf4",metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."})
    bits: int = field(default=4, metadata={"help": "How many bits to use."})
    
    do_train: bool = field(default=True, metadata={"help": 'To train or not to train, that is the question?'})
    do_eval: bool = field(default=True, metadata={"help": 'To train or not to train, that is the question?'})
    output_dir: str = field(default='./output', metadata={"help": 'The output dir for logs and checkpoints'})

    num_train_epochs: float = field(default=3.0, metadata={"help": "total epochs"})
    per_device_train_batch_size: int = field(default=1, metadata={"help": 'The training batch size per GPU. Increase for better speed.'})
    gradient_accumulation_steps: int = field(default=16, metadata={"help": 'How many gradients to accumulate before to perform an optimizer step'})
    dataloader_num_workers: int = field(default=8)

    optim: str = field(default='paged_adamw_32bit', metadata={"help": 'The optimizer to be used, default adamw_torch'})
    learning_rate: float = field(default=0.0002, metadata={"help": 'The learnign rate'})
    lr_scheduler_type: str = field(default='constant', metadata={"help": 'constant | linear | cosine'})
    warmup_ratio: float = field(default=0.03, metadata={"help": 'Fraction of steps to do a warmup for'})
    
    lora_r: int = field(default=64, metadata={"help": "Lora R dimension."})
    lora_alpha: float = field(default=16, metadata={"help": " Lora alpha."})
    lora_dropout: float = field(default=0.0, metadata={"help":"Lora dropout."})
    report_to: str = field(default='none', metadata={'help': "do not use any loggers"})
    remove_unused_columns: bool = field(default=False, metadata={"help": 'Removed unused columns. Needed to make this codebase work.'})
    

@dataclass
class EvaluationArguments:
    checkpoint_dir: Optional[str] = field(default=None)
    full_finetune: bool = field(default=False, metadata={"help": "Finetune the entire model without adapters."})
    

@dataclass
class GenerationArguments:
    # control the length of the output
    max_new_tokens: Optional[int] = field(default=64)
    min_new_tokens : Optional[int] = field(default=1)

    # Generation strategy
    do_sample: Optional[bool] = field(default=True) 
    num_beams: Optional[int] = field(default=1) 
    num_beam_groups: Optional[int] = field(default=1)
    penalty_alpha: Optional[float] = field(default=None)
    use_cache: Optional[bool] = field(default=True)  

    # Hyperparameters for logit manipulation
    temperature: Optional[float] = field(default=1.0)
    top_k: Optional[int] = field(default=50)
    top_p: Optional[float] = field(default=0.9)
    typical_p: Optional[float] = field(default=1.0)
    diversity_penalty: Optional[float] = field(default=0.0)
    repetition_penalty: Optional[float] = field(default=1.0)
    length_penalty: Optional[float] = field(default=1.0)
    no_repeat_ngram_size: Optional[int] = field(default=0)

    num_return_sequences: Optional[int] = field(default=1) 
    output_scores: Optional[bool] = field(default=False)
    return_dict_in_generate: Optional[bool] = field(default=True)

