import os
from typing import List, Optional, Dict, Sequence, Union

import torch
from torch import nn
from transformers import AutoTokenizer, LlamaForCausalLM, GenerationConfig
from transformers.activations import ACT2FN
from peft import PeftModel


class EmbeddingModel(nn.Module):
    def __init__(
            self, 
            embedding_dir: str, 
            input_size: int, 
            intermediate_size: int = 1024,
            output_size: int = 4096, 
            hidden_act: str = 'silu',
    ) -> None:
        super().__init__()
        entity_embedding_path = os.path.join(embedding_dir, 'entity_embeddings.pt')
        query_embedding_path = os.path.join(embedding_dir, 'query_embeddings.pt')

        entity_embeddings = torch.load(entity_embedding_path)
        entity_embeddings.requires_grad = False
        self.ent_embeddings = nn.Embedding.from_pretrained(entity_embeddings)

        query_embeddings = torch.load(query_embedding_path)
        query_embeddings.requires_grad = False
        self.query_embeddings = nn.Embedding.from_pretrained(query_embeddings)
        
        self.adapter = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=intermediate_size, bias=False),
            ACT2FN[hidden_act],
            nn.Linear(in_features=intermediate_size, out_features=output_size, bias=False),
        )
        for layer in self.adapter:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                # torch.nn.init.xavier_normal_(layer.weight)
    
    def forward(self, query_ids, entity_ids):
        """
        Args:
            query_ids: (batch_size, ) 
            entity_ids: (batch_size * K, )
        Returns:
            query_embeds: (batch_size, hidden_size)
            entity_embeds: (batch_size * K, hidden_size)
        """

        query_embeds = self.adapter(self.query_embeddings(query_ids))
        entity_embeds = self.adapter(self.ent_embeddings(entity_ids))
        return query_embeds, entity_embeds


class KGELlama(nn.Module):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        llama_model: Union[LlamaForCausalLM, PeftModel], 
        kge_model: EmbeddingModel,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.llama_model = llama_model
        self.kge_model = kge_model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        query_ids: Optional[torch.LongTensor] = None,
        entity_ids: Optional[torch.LongTensor] = None,
    ):
        """
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            labels: (batch_size, seq_len)
            query_ids: (batch_size, )
            entity_ids: (batch_size, K)
        """
        query_holder = self.tokenizer.convert_tokens_to_ids(['[QUERY]'])[0]
        entity_holder = self.tokenizer.convert_tokens_to_ids(['[ENTITY]'])[0]
        query_position = torch.nonzero(input_ids == query_holder)  # (batch_size, 2)
        entity_position = torch.nonzero(input_ids == entity_holder) # (batch_size*K, 2)

        query_embeds, entity_embeds = self.kge_model(query_ids, entity_ids.view(-1)) 

        input_ids[input_ids == query_holder] = self.tokenizer.pad_token_id
        input_ids[input_ids == entity_holder] = self.tokenizer.pad_token_id
        inputs_embeds = self.llama_model.model.model.embed_tokens(input_ids).clone()

        inputs_embeds[query_position[:, 0], query_position[:, 1]] = query_embeds
        inputs_embeds[entity_position[:, 0], entity_position[:, 1]] = entity_embeds
        
        return self.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
    
    def save_pretrained(self, peft_model_path):
        self.llama_model.save_pretrained(peft_model_path)
        torch.save(self.kge_model.state_dict(), os.path.join(os.path.dirname(peft_model_path), 'kge.bin'))

    def generate(
        self,
        input_ids: torch.LongTensor,
        query_ids: torch.LongTensor,
        entity_ids: torch.LongTensor,
        generation_config: GenerationConfig
    ):
        query_holder = self.tokenizer.convert_tokens_to_ids(['[QUERY]'])[0]
        entity_holder = self.tokenizer.convert_tokens_to_ids(['[ENTITY]'])[0]
        query_position = torch.nonzero(input_ids == query_holder)  # (batch_size, 2)
        entity_position = torch.nonzero(input_ids == entity_holder) # (batch_size*K, 2)

        query_embeds, entity_embeds = self.kge_model(query_ids, entity_ids.view(-1)) 

        input_ids[input_ids == query_holder] = self.tokenizer.pad_token_id
        input_ids[input_ids == entity_holder] = self.tokenizer.pad_token_id
        inputs_embeds = self.llama_model.model.model.embed_tokens(input_ids).clone()

        inputs_embeds[query_position[:, 0], query_position[:, 1]] = query_embeds
        inputs_embeds[entity_position[:, 0], entity_position[:, 1]] = entity_embeds
        
        return self.llama_model.generate(
            inputs_embeds=inputs_embeds,
            generation_config=generation_config,
        )