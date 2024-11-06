import os
import json
import argparse
import random
from copy import deepcopy
from tqdm import tqdm
import numpy as np
import networkx as nx
from collections import Counter, defaultdict
from functools import partial
from multiprocessing import Pool

import torch
from transformers import AutoTokenizer

"""
following methods are used to read textual information and training graph structure
"""

def load_triples(file_path: str):
    triples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            h, r, t = line.strip().split('\t')
            triples.append((h, r, t))
    return triples

def load_text(data_dir: str, tokenizer: AutoTokenizer, max_seq_len: int=50):
    """
    read entity.json and relation.json
    we keep 50 tokens for entity descriptions in FB15K237
    return 3 Dicts: ent2name; ent2desc; rel2name;
    """

    def truncate_text(ent2text: dict, tokenizer: AutoTokenizer, max_len=50):
        ents, texts = [], []
        for k, v in ent2text.items():
            ents.append(k)
            texts.append(v)

        encoded = tokenizer(
            texts, add_special_tokens=False, padding=True, truncation=True, max_length=max_len, 
            return_tensors='pt', return_token_type_ids=False, return_attention_mask=False,
        )
        input_ids = encoded['input_ids']
        truncated_text = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        assert len(ents) == len(truncated_text)

        return {ent: truncated_text[idx] for idx, ent in enumerate(ents)}

    # entity names
    ent2text = json.load(open(os.path.join(data_dir, 'entity.json'), 'r', encoding='utf-8'))
    rel2text = json.load(open(os.path.join(data_dir, 'relation.json'), 'r', encoding='utf-8'))
    ent2name = {k: ent2text[k]['name'] for k in ent2text}
    
    if 'FB15K237' in data_dir:
        # entity descriptions
        desc_path = os.path.join(data_dir, f'desc-{max_seq_len}.json')
        if os.path.exists(desc_path):
            ent2desc = json.load(open(desc_path, 'r', encoding='utf-8'))
        else:
            ent2desc = {k: ent2text[k]['desc'] for k in ent2text}
            ent2desc = truncate_text(ent2desc, tokenizer, max_len=max_seq_len)
            json.dump(ent2desc, open(desc_path, 'w', encoding='utf-8'))

        # relation names
        rel2name = {k: str(k).replace('.', '').replace('_', ' ') for k in rel2text}
    else:
        # eneity descriptions
        ent2desc = {k: ent2text[k]['desc'] for k in ent2text}
        # relation names
        rel2name = {k: rel2text[k]['name'] for k in rel2text}
        
    return ent2name, ent2desc, rel2name

class Relation_Co:
    def __init__(self, file_path) -> None:
        self.triples = self.load_triples(file_path)
        self.rel = self.get_relations()
        self.one_hop_triples, self.one_hop_relations = self.get_one_hop_triples()
        self.rel_co = self.count_rel_co()

    def load_triples(self, file_path: str, format='hrt'):
        triples = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                if format == 'hrt':
                    h, r, t = line.strip().split('\t')
                elif format == 'htr':
                    h, t, r = list(map(int, line.strip().split(' ')))
                else:
                    raise NotImplementedError()
                triples.append((h, r, t))
        return triples

    def get_relations(self):
        rel = set()
        for h, r, t in self.triples:
            rel.add(r)
        return rel
    
    def get_one_hop_triples(self):
        one_hop_triples = defaultdict(set)
        one_hop_relations = defaultdict(set)
        for h, r, t in self.triples:
            one_hop_triples[h].add((h, r, t))
            one_hop_triples[t].add((h, r, t))
            one_hop_relations[h].add((r, 0))
            one_hop_relations[t].add((r, 1))
        return one_hop_triples, one_hop_relations

    def count_rel_co(self):
        # tail_prediction_co = defaultdict(int)
        # head_prediction_co = defaultdict(int)
        rel_co = defaultdict(int)
        for entity, one_hop_triples in self.one_hop_triples.items():
            for h, r, t in one_hop_triples:
                for r_sample, direct in self.one_hop_relations[entity]:
                    if r == r_sample:
                        continue
                    elif h == entity:
                        rel_co[((r, 0) , (r_sample, direct))] += 1
                    else:
                        rel_co[((r, 1) , (r_sample, direct))] += 1
        return rel_co

    def get_rel_co(self, rel, direct):
        for r in self.rel:
            if r != rel:
                print(r, self.rel_co[(rel, direct), (r, 0)], self.rel_co[(rel, direct), (r, 1)])

class KnowledgeGraph:
    def __init__(self, args) -> None:
        self.args = args

        # textural information
        self.ent2name, self.ent2desc, self.rel2name = load_text(args.data_dir, tokenizer, args.max_seq_len)
        self.idx2ent = {idx: ent for idx, ent in enumerate(self.ent2name.keys())}
        self.ent2idx = {ent: idx for idx, ent in self.idx2ent.items()}

        # triplets
        self.train_triplets = load_triples(os.path.join(args.data_dir, 'train.txt'))
        self.valid_triplets = load_triples(os.path.join(args.data_dir, 'valid.txt'))
        self.test_triplets = load_triples(os.path.join(args.data_dir, 'test.txt'))
        # all entities and all relations
        triplets = self.train_triplets
        self.ent_list = sorted(list(set([h for h, _, _ in triplets] + [t for _, _, t in triplets])))
        self.rel_list = sorted(list(set([r for _, r, _ in triplets])))
        print(f'entity num: {len(self.ent_list)}; relation num: {len(self.rel_list)}')

        self.relation_co = Relation_Co(os.path.join(args.data_dir, 'train.txt'))

        self.graph = nx.MultiDiGraph()
        for h, r, t in self.train_triplets:
            self.graph.add_edge(h, t, relation=r)
        print(self.graph)

    def all_shortest_paths(self, fact: dict):
        try:
            paths = nx.all_shortest_paths(self.graph, fact['h'], fact['t'])
            return [len(path)-1 for path in paths]
        except:
            return []
    
    def neighbors_condition(self, ent, rel, direct):
        out_edges = []
        score_out = []
        for h, t, attr_dict in self.graph.out_edges(ent, data=True):
            assert ent == h
            out_edges.append((h, attr_dict['relation'], t))
            score_out.append(self.relation_co.rel_co[(rel, direct), (attr_dict['relation'], 0)])
        out_sorted_indices_desc = np.argsort(score_out)[::-1]

        in_edges = []
        score_in = []
        for h, t, attr_dict in self.graph.in_edges(ent, data=True):
            assert ent == t
            in_edges.append((h, attr_dict['relation'], t))
            score_in.append(self.relation_co.rel_co[(rel, direct), (attr_dict['relation'], 1)])
        in_sorted_indices_desc = np.argsort(score_in)[::-1]

        if self.args.neighbor_num <= len(out_edges):
            return [out_edges[out_sorted_indices_desc[i]] for i in range(self.args.neighbor_num)]
        elif self.args.neighbor_num <= len(out_edges + in_edges):
            return out_edges + [in_edges[in_sorted_indices_desc[i]] for i in range(self.args.neighbor_num - len(out_edges))]
        else:
            edges = out_edges + in_edges
            random.shuffle(edges)
            return edges

    def neighbors(self, ent):
        out_edges = []
        for h, t, attr_dict in self.graph.out_edges(ent, data=True):
            assert ent == h
            out_edges.append((h, attr_dict['relation'], t))
        
        in_edges = []
        for h, t, attr_dict in self.graph.in_edges(ent, data=True):
            assert ent == t
            in_edges.append((h, attr_dict['relation'], t))
        
        if self.args.neighbor_num <= len(out_edges):
            return random.sample(out_edges, self.args.neighbor_num)
        elif self.args.neighbor_num <= len(out_edges + in_edges):
            return random.sample(out_edges + in_edges, self.args.neighbor_num)
        else:
            edges = out_edges + in_edges
            random.shuffle(edges)
            return edges


"""
following methods are used to preprocess the outputs from different KGE models
"""

def TransE_preprocess(args, graph: KnowledgeGraph):
    """
    Preprocess the output from TransE
    We need to prepare query_embeddings, entity_embeddings, and train/valid/test set for DIFT
    """
    def load_triplets_with_ids(file_path: str):
        triplets = []
        with open(file_path, 'r', encoding='utf-8') as fin:
            lines = fin.readlines()
            data_num = int(lines[0].strip())
            for line in lines[1: ]:
                h, t, r = line.strip().split(' ')
                triplets.append((int(h), int(r), int(t)))
            assert data_num == len(triplets), f'{data_num}\t{len(triplets)}'
        return triplets
    
    def load_ent_or_rel_to_id(file_path: str):
        ent2idx = dict()
        idx2ent = dict()
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            num = int(str(lines[0]).strip())
            for line in lines[1:]:
                ent, idx = line.strip().split('\t')
                idx = int(idx)
                ent2idx[ent] = idx
                idx2ent[idx] = ent
            assert len(ent2idx) == num
        return ent2idx, idx2ent
    
    TransE_dir = os.path.join(args.data_dir, args.kge_model)
    ent2name = graph.ent2name
    
    valid_triplets = load_triplets_with_ids(os.path.join(TransE_dir, 'valid2id.txt'))
    test_tripelts = load_triplets_with_ids(os.path.join(TransE_dir, 'test2id.txt'))
    assert len(valid_triplets) == len(graph.valid_triplets)
    assert len(test_tripelts) == len(graph.test_triplets)

    ent2idx, idx2ent = load_ent_or_rel_to_id(os.path.join(TransE_dir, 'entity2id.txt'))
    rel2idx, idx2rel = load_ent_or_rel_to_id(os.path.join(TransE_dir, 'relation2id.txt'))
    assert len(idx2ent) == len(graph.idx2ent)
    assert len(idx2rel) == len(graph.rel2name)

    entity_embeddings_path = os.path.join(TransE_dir, 'entity_embeddings.pt')
    if not os.path.exists(entity_embeddings_path):
        ent_embeds = torch.from_numpy(np.load(os.path.join(TransE_dir, 'embeds_ent.npy')))
        assert ent_embeds.shape[0] == len(graph.idx2ent)
        new_ent_embeds = torch.zeros_like(ent_embeds)
        for idx in range(ent_embeds.shape[0]):
            ent = idx2ent[idx]
            new_ent_embeds[graph.ent2idx[ent]] = ent_embeds[idx]
        assert new_ent_embeds.shape[0] == len(graph.ent2idx)
        torch.save(new_ent_embeds, entity_embeddings_path)
    
    query_embeddings_path = os.path.join(TransE_dir, 'query_embeddings.pt')
    if not os.path.exists(query_embeddings_path):
        h_embeds = torch.from_numpy(np.load(os.path.join(TransE_dir, 'embed_h.npy')))
        r_embeds = torch.from_numpy(np.load(os.path.join(TransE_dir, 'embed_r.npy')))
        t_embeds = torch.from_numpy(np.load(os.path.join(TransE_dir, 'embed_t.npy')))
        triplets = valid_triplets + test_tripelts
        assert h_embeds.shape[0] == r_embeds.shape[0] == t_embeds.shape[0] == len(triplets)
        
        query_embeddings = torch.zeros(2*len(triplets), h_embeds.shape[-1])
        idx = 0
        for i in range(len(triplets)):
            query_embeddings[idx] = t_embeds[i] - r_embeds[i]
            query_embeddings[idx+1] = h_embeds[i] + r_embeds[i]
            idx += 2
        torch.save(query_embeddings, query_embeddings_path)
    query_embeddings = torch.load(query_embeddings_path, map_location='cpu')

    ent_embeds = torch.from_numpy(np.load(os.path.join(TransE_dir, 'embeds_ent.npy')))
    rel_embeds = torch.from_numpy(np.load(os.path.join(TransE_dir, 'embeds_rel.npy')))

    head_ranks = np.load(os.path.join(TransE_dir, 'rank_head.npy'))
    head_topks = np.load(os.path.join(TransE_dir, 'topk_head.npy'))
    head_topks_scores = np.load(os.path.join(TransE_dir, 'topk_head_scores.npy'))
    tail_ranks = np.load(os.path.join(TransE_dir, 'rank_tail.npy'))
    tail_topks = np.load(os.path.join(TransE_dir, 'topk_tail.npy'))
    tail_topks_scores = np.load(os.path.join(TransE_dir, 'topk_tail_scores.npy'))

    data = []
    triplets = valid_triplets + test_tripelts
    for idx, (h, r, t) in enumerate(graph.valid_triplets + graph.test_triplets):
        h_idx, r_idx, t_idx = triplets[idx]
        assert all(query_embeddings[2*idx] == ent_embeds[t_idx]-rel_embeds[r_idx])
        assert all(query_embeddings[2*idx+1] == ent_embeds[h_idx]+rel_embeds[r_idx])

        tail_topk = [idx2ent[e_idx] for e_idx in tail_topks[idx].tolist()][: args.topk]
        tail_topk_scores = [score * 1e-5 for score in tail_topks_scores[idx].tolist()[: args.topk]]
        tail_rank = int(tail_ranks[idx])
        tail_topk_names = [ent2name[ent] for ent in tail_topk]
        tail_entity_ids = [graph.ent2idx[ent] for ent in tail_topk]
        
        head_topk = [idx2ent[e_idx] for e_idx in head_topks[idx].tolist()][: args.topk]
        head_topk_scores = [score * 1e-5 for score in head_topks_scores[idx].tolist()[: args.topk]]
        head_rank = int(head_ranks[idx])
        head_topk_names = [ent2name[ent] for ent in head_topk]
        head_entity_ids = [graph.ent2idx[ent] for ent in head_topk]
        
        head_prediction = {
            'triplet': (t, r, h),
            'inverse': True,
            'topk_ents': head_topk,
            'topk_names': head_topk_names,
            'topk_scores': head_topk_scores,
            'rank': head_rank,
            'query_id': 2*idx,
            'entity_ids': head_entity_ids
        }

        tail_prediction = {
            'triplet': (h, r, t),
            'inverse': False,
            'topk_ents': tail_topk,
            'topk_names': tail_topk_names,
            'topk_scores': tail_topk_scores,
            'rank': tail_rank,
            'query_id': 2*idx+1,
            'entity_ids': tail_entity_ids
        }
        
        data.append(tail_prediction)
        data.append(head_prediction)
    
    valid_output = data[: len(valid_triplets)*2]
    test_output = data[len(valid_triplets)*2: ]
    assert len(graph.valid_triplets) == len(valid_output) // 2
    assert len(graph.test_triplets) == len(test_output) // 2
    return valid_output, test_output
 

def SimKGC_preprocess(args, graph: KnowledgeGraph):
    """
    Args:
        SimKGC提供的文件: 
        entities.txt: store the orders of entities in SimKGC, reorder simkgc_entity_embeddings.pt
        valid_query_embeddings.pt, test_query_embeddings.pt: merge as query_embeddings.pt
        valid.json, test.json: head prediction and tail prediction results
    """
    SimKGC_dir = os.path.join(args.data_dir, args.kge_model)
    if args.dataset == 'Wikidata5M-ind':
        with open(os.path.join(SimKGC_dir, 'entities.txt'), 'r', encoding='utf-8') as f:
            ents = [line.strip() for line in f.readlines()]
            graph.ent2idx = {ent: idx for idx, ent in enumerate(ents)}
            graph.idx2ent = {idx: ent for idx, ent in enumerate(ents)}

    # 将simkgc_entity_embeddings.pt存储的表示调整顺序, 存储为entity_embeddings.py
    entity_embeddings_path = os.path.join(SimKGC_dir, 'entity_embeddings.pt')
    if not os.path.exists(entity_embeddings_path):
        print('reorder entity embeddings')
        simkgc_ent_embeds  = torch.load(os.path.join(SimKGC_dir, 'simkgc_entity_embeddings.pt'), map_location='cpu')
        ents = []
        with open(os.path.join(SimKGC_dir, 'entities.txt'), 'r', encoding='utf-8') as fin:
            for line in fin.readlines():
                ent = line.strip()
                ents.append(ent)
        entity_embeddings = torch.zeros_like(simkgc_ent_embeds)
        for idx, ent in enumerate(ents):
            entity_embeddings[graph.ent2idx[ent]] = simkgc_ent_embeds[idx]
        assert entity_embeddings.shape[0] == len(graph.ent2idx)
        torch.save(entity_embeddings, entity_embeddings_path)

    # valid_query_embeddings 和 test_query_embeddings 直接合并
    query_embeddings_path = os.path.join(SimKGC_dir, 'query_embeddings.pt')
    if not os.path.exists(query_embeddings_path):
        print('merge query embeddings')
        valid_query_embeds = torch.load(os.path.join(SimKGC_dir, 'valid_query_embeddings.pt'), map_location='cpu')
        test_query_embeds = torch.load(os.path.join(SimKGC_dir, 'test_query_embeddings.pt'), map_location='cpu')
        assert valid_query_embeds.shape[0] == len(graph.valid_triplets) *2
        assert test_query_embeds.shape[0] == len(graph.test_triplets) * 2
        query_embeds = torch.cat([valid_query_embeds, test_query_embeds], dim=0)
        torch.save(query_embeds, query_embeddings_path)
    query_embeds = torch.load(query_embeddings_path, map_location='cpu')

    # xxx_input: triplet, rank, top-50, top-50 scores
    # head prediction and tail prediction
    valid_input = json.load(open(os.path.join(SimKGC_dir, 'valid.json'), 'r', encoding='utf-8'))
    test_input = json.load(open(os.path.join(SimKGC_dir, 'test.json'), 'r', encoding='utf-8'))
    input_data = valid_input + test_input
    triplets = graph.valid_triplets + graph.test_triplets

    assert len(triplets) == len(input_data) // 2
    assert len(input_data) == query_embeds.shape[0]
    output_data = []
    for idx, (h, r, t) in enumerate(triplets):
        # head prediction
        input1 = input_data[2*idx]
        t1, r1, h1 = input1['head'], input1['relation'], input1['tail']
        head_rank = input1['rank']
        head_topk = input1['topk'][: args.topk]
        head_topk_scores = input1['topk_scores'][: args.topk]
        # tail prediction
        input2 = input_data[2*idx+1]
        h2, r2, t2 = input2['head'], input2['relation'], input2['tail']
        tail_rank = input2['rank']
        tail_topk = input2['topk'][: args.topk]
        tail_topk_scores = input2['topk_scores'][: args.topk]

        assert h == h1 and h == h2 and r1[len('inverse '):]==r2 and t == t1 and t == t2

        # find entity and entity name by idx
        head_topk_idxs = [graph.ent2idx[ent] for ent in head_topk]
        head_topk_names = [graph.ent2name[ent] for ent in head_topk]
    
        tail_topk_idxs = [graph.ent2idx[ent] for ent in tail_topk]
        tail_topk_names = [graph.ent2name[ent] for ent in tail_topk]
        # check
        if tail_rank <= len(tail_topk):
            assert t == tail_topk[tail_rank-1]
        if head_rank <= len(head_topk):
            assert h == head_topk[head_rank-1]

        head_prediction = {
            'triplet': (t, r, h),
            'inverse': True,
            'topk_ents': head_topk,
            'topk_names': head_topk_names,
            'topk_scores': head_topk_scores,
            'rank': head_rank,
            'query_id': 2*idx,
            'entity_ids': head_topk_idxs
        }

        tail_prediction = {
            'triplet': (h, r, t),
            'inverse': False,
            'rank': tail_rank,
            'topk_ents': tail_topk,
            'topk_names': tail_topk_names,
            'topk_scores': tail_topk_scores,
            'query_id': 2*idx+1,
            'entity_ids': tail_topk_idxs,
        }
        
        output_data.append(tail_prediction)
        output_data.append(head_prediction)
    
    valid_output = output_data[: len(valid_input)]
    test_output = output_data[len(valid_input): ]
    assert len(graph.valid_triplets) == len(valid_output) // 2
    assert len(graph.test_triplets) == len(test_output) // 2
    return valid_output, test_output


def CoLE_preprocess(args, graph: KnowledgeGraph):
    """
    CoLE提供的数据包括:
    entity_embeddings.pt: 所有实体的表示, 顺序按照entity.json中的顺序
    relation_embeddings.pt: 所有关系的表示, 顺序按照relation.json的顺序
    valid_query_embeddings.pt: 验证集的查询表示, 顺序按照验证集中三元组的顺序, 第一个是head prediction, 第二个是tail prediction
    test_query_embeddings.pt: 测试集的查询表示, 顺序同上
    valid.json, test.json: 输出结果, 包括triplet, rank, top-50, top-50 scores
    """
    CoLE_dir = os.path.join(args.data_dir, args.kge_model)

    # 合并验证集和测试集的query embeddings
    query_embeddings_path = os.path.join(CoLE_dir, 'query_embeddings.pt')
    if not os.path.exists(query_embeddings_path):
        valid_query_embeds = torch.load(os.path.join(CoLE_dir, 'valid_query_embeddings.pt'), map_location='cpu')
        test_query_embeds = torch.load(os.path.join(CoLE_dir, 'test_query_embeddings.pt'), map_location='cpu')
        query_embeds = torch.cat([valid_query_embeds, test_query_embeds], dim=0)
        torch.save(query_embeds, query_embeddings_path)
    query_embeds = torch.load(query_embeddings_path, map_location='cpu')

    # json文件是字典构成的列表, 每个字典的key为: triplet, rank, top-50, top-50 scores
    # head prediction and tail prediction
    valid_input = json.load(open(os.path.join(CoLE_dir, 'valid.json'), 'r', encoding='utf-8'))
    test_input = json.load(open(os.path.join(CoLE_dir, 'test.json'), 'r', encoding='utf-8'))
    input_data = valid_input + test_input
    triplets = graph.valid_triplets + graph.test_triplets

    assert len(triplets) == len(input_data) // 2
    assert len(input_data) == query_embeds.shape[0]
    output_data = []
    for idx, (h, r, t) in enumerate(triplets):
        # head prediction
        input1 = input_data[2*idx]
        t1, r1, h1 = input1['triplet']
        head_rank = input1['rank']
        head_topk_idxs = input1['top-50'][: args.topk]
        head_topk_scores = input1['top-50 scores'][: args.topk]
        # tail prediction
        input2 = input_data[2*idx+1]
        h2, r2, t2 = input2['triplet']
        tail_rank = input2['rank']
        tail_topk_idxs = input2['top-50'][: args.topk]
        tail_topk_scores = input2['top-50 scores'][: args.topk]
        # 确保相邻两个样本是同一个三元组的head tail prediction
        assert h == h1 and h == h2 and r == r1 and r == r2 and t == t1 and t == t2

        # top-50存储的是实体对应的下标, 顺序由entity.json确定
        # find entity and entity name by idx
        head_topk = [graph.idx2ent[e_idx] for e_idx in head_topk_idxs]
        head_topk_names = [graph.ent2name[ent] for ent in head_topk]
    
        tail_topk = [graph.idx2ent[e_idx] for e_idx in tail_topk_idxs]
        tail_topk_names = [graph.ent2name[ent] for ent in tail_topk]
        # check
        if tail_rank <= len(tail_topk):
            assert t == tail_topk[tail_rank-1]
        if head_rank <= len(head_topk):
            assert h == head_topk[head_rank-1]

        head_prediction = {
            'triplet': (t, r, h),
            'inverse': True,
            'rank': head_rank,
            'topk_ents': head_topk,
            'topk_names': head_topk_names,
            'topk_scores': head_topk_scores,
            'query_id': 2*idx,
            'entity_ids': head_topk_idxs,
        }

        tail_prediction = {
            'triplet': (h, r, t),
            'inverse': False,
            'rank': tail_rank,
            'topk_ents': tail_topk,
            'topk_names': tail_topk_names,
            'topk_scores': tail_topk_scores,
            'query_id': 2*idx+1,
            'entity_ids': tail_topk_idxs,
        }
        
        output_data.append(tail_prediction)
        output_data.append(head_prediction)
    
    valid_output = output_data[: len(valid_input)]
    test_output = output_data[len(valid_input): ]
    assert len(graph.valid_triplets) == len(valid_output) // 2
    assert len(graph.test_triplets) == len(test_output) // 2
    return valid_output, test_output


"""
following methods are used to construct train/valid/test datasets
"""

def divide_valid(args, data: list):
    """
    filter the valid data, and further divide it to a train dataset and a valid dataset
    """
    # 9:1 = train: valid
    random.shuffle(data)
    valid_data = data[: int(len(data) * 0.1)]
    train_data = data[int(len(data) * 0.1) :]

    # compute the confidence score
    score_list = []
    for item in train_data:
        if item['rank'] <= args.topk:
            score_list.append(100 * item['topk_scores'][item['rank'] - 1] + 1 /item['rank'])
        else:
            score_list.append(1 /item['rank'])
    # set threshold to filter out samples
    weights = np.array(score_list)
    threshold = args.threshold
    indices = np.where(weights > threshold)[0]
    print('keeped train', len(indices) / len(train_data))

    new_train = []
    count = 0
    for i in range(len(train_data)):
        if i in indices:
            new_train.append(train_data[i])
            if train_data[i]['rank'] <= args.topk:
                count += 1
    print(f'train: {len(new_train)}; valid: {len(valid_data)}')
    return new_train, valid_data


def make_prompt(input_data, graph: KnowledgeGraph):
    """
    input_data是一个字典: {triplet, inverse, rank, topk_ents, topk_names, topk_scores, query_id, entity_ids}
    其中query_id, entity_ids是知识注入时所需要的额外数据
    需要填充的数据包括input, output
    """
    args = graph.args
    
    tail_prediction = not input_data['inverse']
    if tail_prediction:
        h, r, t = input_data['triplet']
    else:
        t, r, h = input_data['triplet']
    ent2name, ent2desc, rel2name = graph.ent2name, graph.ent2desc, graph.rel2name
    h_name, h_desc = ent2name[h], ent2desc[h]
    r_name = rel2name[r]
    t_name, t_desc = ent2name[t], ent2desc[t]

    if args.shuffle_candidates:
        topk_ents = input_data['topk_ents']
        choices = deepcopy(topk_ents)
        random.shuffle(choices)
        entity_ids = [graph.ent2idx[ent] for ent in choices]
        input_data['entity_ids'] = entity_ids
        choices = [graph.ent2name[ent] for ent in choices]
    else:
        choices = input_data['topk_names']
    input_data['choices'] = choices
    if args.add_special_tokens:
        try:
            choices = [ent_name + ' [ENTITY]' for ent_name in choices]
        except:
            print(input_data)
            print(choices)
            exit(0)
    choices = '[' + '; '.join(choices) + ']'
    
    if tail_prediction:
        if args.add_special_tokens:
            prompt = f'Here is a triplet with tail entity t unknown: ({h_name}, {r_name}, t [QUERY]).\n\n'
        else:
            prompt = f'Here is a triplet with tail entity t unknown: ({h_name}, {r_name}, t).\n\n'
        if args.add_entity_desc:
            prompt += f'Following are some details about {h_name}:\n{h_desc}\n\n'
        if args.add_neighbors:
            if args.condition_neighbors:
                neighbors = [(ent2name[e1], rel2name[r1], ent2name[e2]) for e1, r1, e2 in graph.neighbors_condition(h, r, 0)]
            else:
                neighbors = [(ent2name[e1], rel2name[r1], ent2name[e2]) for e1, r1, e2 in graph.neighbors(h)]
            neighbors = '[' + '; '.join([f'({e1}, {r1}, {e2})' for e1, r1, e2 in neighbors]) + ']'
            prompt += f'Following are some triplets about {h_name}:\n{neighbors}\n\n'
        prompt += f'What is the entity name of t? Select one from the list: {choices}\n\n[Answer]: '
        
        input_data['input'] = prompt
        input_data['output'] = t_name
    else:  # head prediction
        if args.add_special_tokens:
            prompt = f'Here is a triplet with head entity h unknown: (h [QUERY], {r_name}, {t_name}).\n\n'
        else:
            prompt = f'Here is a triplet with head entity h unknown: (h, {r_name}, {t_name}).\n\n'
        if args.add_entity_desc:
            prompt += f'Following are some details about {t_name}:\n{t_desc}\n\n'
        if args.add_neighbors:
            if args.condition_neighbors:
                neighbors = [(ent2name[e1], rel2name[r1], ent2name[e2]) for e1, r1, e2 in graph.neighbors_condition(t, r, 1)]
            else:
                neighbors = [(ent2name[e1], rel2name[r1], ent2name[e2]) for e1, r1, e2 in graph.neighbors(t)]
            neighbors = '[' + '; '.join([f'({e1}, {r1}, {e2})' for e1, r1, e2 in neighbors]) + ']'
            prompt += f'Following are some triplets about {t_name}:\n{neighbors}\n\n'
        prompt += f'What is the entity name of h? Select one from the list: {choices}\n\n[Answer]: '

        input_data['input'] = prompt
        input_data['output'] = h_name
    
    return input_data


def make_dataset_mp(data: list, output_file: str):
    """
    construct the dataset with multi-processing
    """
    with Pool(20) as p:
        data = p.map(partial(make_prompt, graph=graph), data)
        
    json.dump(data, open(output_file, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--llm_dir', type=str, default="llama2-7b-chat-hf", help="put your own LLM folder here")

    parser.add_argument('--dataset', type=str, default='FB15K237', help='FB15K237 | WN18RR')
    parser.add_argument('--topk', type=int, default=20, help='number of candidates')
    parser.add_argument('--threshold', type=float, default=0.05, help='threshold for truncated sampling')
    parser.add_argument('--kge_model', type=str, default='SimKGC', help='TransE | SimKGC | CoLE')
    parser.add_argument('--output_folder', type=str, default='data_top10', help='output folder for dataset')
    parser.add_argument('--add_special_tokens', type=bool, default=True, help='add place holder for knowledge injection')
    parser.add_argument('--add_entity_desc', type=bool, default=True)
    parser.add_argument('--max_seq_len', type=int, default=50, help='desc of FB15K237 is too long')
    parser.add_argument('--add_neighbors', type=bool, default=True)
    parser.add_argument('--condition_neighbors', type=bool, default=True, help="random or heuristic")
    parser.add_argument('--neighbor_num', type=int, default=10)
    parser.add_argument('--shuffle_candidates', type=bool, default=False, help="whether shuffle the candidates for analyses")
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    # add data_dir and output_dir
    args.data_dir = os.path.join(os.path.dirname(__file__), args.dataset)
    args.output_dir = os.path.join(args.data_dir, args.kge_model, args.output_folder)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.llm_dir, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    
    graph = KnowledgeGraph(args)

    if args.kge_model == 'TransE':
        valid_data, test_data = TransE_preprocess(args, graph)
    elif args.kge_model == 'SimKGC':
        valid_data, test_data = SimKGC_preprocess(args, graph)
    elif args.kge_model == 'CoLE':
        valid_data, test_data = CoLE_preprocess(args, graph)
    else:
        raise NotImplementedError()
    
    llm_train, llm_valid = divide_valid(args, valid_data)
    train_examples = make_dataset_mp(llm_train, os.path.join(args.output_dir, 'train.json'))
    valid_examples = make_dataset_mp(llm_valid, os.path.join(args.output_dir, 'valid.json'))
    test_examples = make_dataset_mp(test_data, os.path.join(args.output_dir, 'test.json'))

    train_examples = json.load(open(os.path.join(args.output_dir, 'train.json'), 'r', encoding='utf-8'))
    valid_examples = json.load(open(os.path.join(args.output_dir, 'valid.json'), 'r', encoding='utf-8'))
    test_examples = json.load(open(os.path.join(args.output_dir, 'test.json'), 'r', encoding='utf-8'))

    args = vars(args)
    # statistics
    args['train_num'] = len(train_examples)
    args['valid_num'] = len(valid_examples)
    args['test_num'] = len(test_examples)
    # KGE model metrics
    kge_ranks = np.array([example['rank'] for example in test_examples])
    args['hits@1'] = np.round(np.mean(kge_ranks <= 1), 2)
    args['hits@3'] = np.round(np.mean(kge_ranks <= 3), 2)
    args['hits@10'] = np.round(np.mean(kge_ranks <= 10), 2)
    args['mrr'] = np.round(np.mean( 1.0 / kge_ranks), 3)

    # avg nums of input token
    texts = [data['input'] for data in train_examples + valid_examples + test_examples]
    encoded = tokenizer(texts, add_special_tokens=False)
    lens = [len(input_ids) for input_ids in encoded['input_ids']]
    args['min_seq_len'] = int(np.min(lens))
    args['max_seq_len'] = int(np.max(lens))
    args['avg_seq_len'] = int(np.round(np.mean(lens)))
    
    with open(os.path.join(args['output_dir'], 'args.txt'), 'w', encoding='utf-8') as f:
        for key, value in args.items():
            f.write(f'{key}: {value}\n')
    
