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
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa)."""
#
#
from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random
from collections import defaultdict
import re
import shutil

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange
import time
from transformers import (WEIGHTS_NAME, BertConfig,
                                  BertTokenizer,
                                  RobertaConfig,
                                  RobertaTokenizer,
                                  get_linear_schedule_with_warmup,
                                  AdamW,
                                  BertForACEBothOneDropoutSub,
                                  BertForACEBothOneDropoutSub_1,
                                  BertForACEBothOneDropoutSub_1_span,
                                  AlbertForACEBothSub,
                                  AlbertConfig,
                                  AlbertTokenizer,
                                  AlbertForACEBothOneDropoutSub,
                                  AlbertForACEBothOneDropoutSub_1,
                                  BertForACEBothOneDropoutSubNoNer,
                                  RobertForACEBothOneDropoutSpanSub,
BertForACEBothOneDropoutSpanSub,
                                  )

from transformers import AutoTokenizer
from torch.utils.data import TensorDataset, Dataset
import json
import pickle
import numpy as np
import unicodedata
import itertools
import timeit

from tqdm import tqdm

logger = logging.getLogger(__name__)


ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig,  AlbertConfig)), ())

MODEL_CLASSES = {
    'bertsub': (BertConfig, BertForACEBothOneDropoutSub, BertTokenizer),
    'bertsub_1': (BertConfig, BertForACEBothOneDropoutSub_1, BertTokenizer),
    'bertsub_1_span': (BertConfig, BertForACEBothOneDropoutSub_1_span, BertTokenizer),
    'bertnonersub': (BertConfig, BertForACEBothOneDropoutSubNoNer, BertTokenizer),
    'albertsub': (AlbertConfig, AlbertForACEBothOneDropoutSub, AlbertTokenizer),
    'albertsub_1': (AlbertConfig, AlbertForACEBothOneDropoutSub_1, AlbertTokenizer),
    'robertsub':(RobertaConfig, RobertForACEBothOneDropoutSpanSub, RobertaTokenizer),
    'bertsub_span':(BertConfig, BertForACEBothOneDropoutSpanSub, BertTokenizer),
}

task_ner_labels = {
    'ace04': ['FAC', 'WEA', 'LOC', 'VEH', 'GPE', 'ORG', 'PER'],
    'ace05': ['FAC', 'WEA', 'LOC', 'VEH', 'GPE', 'ORG', 'PER'],
    'scierc': ['Method', 'OtherScientificTerm', 'Task', 'Generic', 'Material', 'Metric'],
}

task_rel_labels = {
    'ace04': ['PER-SOC', 'OTHER-AFF', 'ART', 'GPE-AFF', 'EMP-ORG', 'PHYS'],
    'ace05': ['PER-SOC', 'ART', 'ORG-AFF', 'GEN-AFF', 'PHYS', 'PART-WHOLE'],
    'scierc': ['PART-OF', 'USED-FOR', 'FEATURE-OF', 'CONJUNCTION', 'EVALUATE-FOR', 'HYPONYM-OF', 'COMPARE'],
}
data_dir={'data_path_loss':'train_re_span_loss.txt','data_name_loss':'train_re_span_loss','data_path_f1':'train_re_span_f1.txt','data_name_f1':'train_re_span_f1'}


class ACEDataset(Dataset):
    def __init__(self, tokenizer, args=None, evaluate=False, do_test=False, max_pair_length=None):

        if not evaluate:
            file_path = os.path.join(args.data_dir, args.train_file)#re阶段所用的数据集和ner阶段是一样的
        else:
            if do_test:#使用的是NER模型的预测输出
                if args.test_file.find('models')==-1:
                    file_path = os.path.join(args.data_dir, args.test_file)
                else:
                    file_path = args.test_file
            else:
                if args.dev_file.find('models')==-1:
                    file_path = os.path.join(args.data_dir, args.dev_file)
                else:
                    file_path = args.dev_file

        assert os.path.isfile(file_path)

        self.file_path = file_path
                
        self.tokenizer = tokenizer
        self.max_seq_length = args.max_seq_length#256
        self.max_pair_length = max_pair_length#16
        self.max_entity_length = self.max_pair_length*2#32

        self.evaluate = evaluate
        self.use_typemarker = args.use_typemarker#不使用类型标记
        self.local_rank = args.local_rank
        self.args = args
        self.model_type = args.model_type
        self.no_sym = args.no_sym

        if args.data_dir.find('ace05')!=-1:
            self.ner_label_list = ['NIL', 'FAC', 'WEA', 'LOC', 'VEH', 'GPE', 'ORG', 'PER']

            if args.no_sym:
                label_list = ['PER-SOC', 'ART', 'ORG-AFF', 'GEN-AFF', 'PHYS', 'PART-WHOLE']
                self.sym_labels = ['NIL']
                self.label_list = self.sym_labels + label_list
            else:
                label_list = ['ART', 'ORG-AFF', 'GEN-AFF', 'PHYS',  'PART-WHOLE']
                self.sym_labels = ['NIL', 'PER-SOC']
                self.label_list = self.sym_labels + label_list

        elif args.data_dir.find('ace04')!=-1:
            self.ner_label_list = ['NIL', 'FAC', 'WEA', 'LOC', 'VEH', 'GPE', 'ORG', 'PER']

            if args.no_sym:
                label_list = ['PER-SOC', 'OTHER-AFF', 'ART', 'GPE-AFF', 'EMP-ORG', 'PHYS']
                self.sym_labels = ['NIL']
                self.label_list = self.sym_labels + label_list
            else:
                label_list = ['OTHER-AFF', 'ART', 'GPE-AFF', 'EMP-ORG', 'PHYS']
                self.sym_labels = ['NIL', 'PER-SOC']
                self.label_list = self.sym_labels + label_list

        elif args.data_dir.find('scierc')!=-1:      
            self.ner_label_list = ['NIL', 'Method', 'OtherScientificTerm', 'Task', 'Generic', 'Material', 'Metric']

            if args.no_sym:
                label_list = ['CONJUNCTION', 'COMPARE', 'PART-OF', 'USED-FOR', 'FEATURE-OF',  'EVALUATE-FOR', 'HYPONYM-OF']
                self.sym_labels = ['NIL']
                self.label_list = self.sym_labels + label_list
            else:
                label_list = ['PART-OF', 'USED-FOR', 'FEATURE-OF',  'EVALUATE-FOR', 'HYPONYM-OF']
                self.sym_labels = ['NIL', 'CONJUNCTION', 'COMPARE']#连接，比较，这个是对称关系
                self.label_list = self.sym_labels + label_list

        else:
            assert (False)  

        self.global_predicted_ners = {}
        self.initialize()
 
    def initialize(self):
        tokenizer = self.tokenizer
        vocab_size = tokenizer.vocab_size#31090
        max_num_subwords = self.max_seq_length - 4  # for two marker  252
        label_map = {label: i for i, label in enumerate(self.label_list)}
        ner_label_map = {label: i for i, label in enumerate(self.ner_label_list)}

        def tokenize_word(text):
            if (
                isinstance(tokenizer, RobertaTokenizer)
                and (text[0] != "'")
                and (len(text) != 1 or not self.is_punctuation(text))
            ):
                return tokenizer.tokenize(text, add_prefix_space=True)
            return tokenizer.tokenize(text)

        f = open(self.file_path, "r", encoding='utf-8')
        self.ner_tot_recall = 0
        self.tot_recall = 0
        self.data = []
        self.ner_golden_labels = set([])
        self.golden_labels = set([])
        self.golden_labels_withner = set([])
        maxR = 0
        maxL = 0
        for l_idx, line in enumerate(f):#每次读一行sentences，包括很多个句子
            data = json.loads(line)

            if self.args.output_dir.find('test')!=-1:
                if len(self.data) > 100:
                    break

            sentences = data['sentences']#一个sentences里面含有几个sentence
            if 'predicted_ner' in data:       # e2e predict
               ners = data['predicted_ner']               
            else:
               ners = data['ner']#这些位置信息都是一个sentences数据中的位置，但是由于列表的性质，每个对应的列表元素都是一个单独的sentence，包括对应的ner，re[0]对应的位置也都是这样

            std_ners = data['ner']

            relations = data['relations']

            for sentence_relation in relations:
                for x in sentence_relation:
                    if x[4] in self.sym_labels[1:]:#如果是对称关系，则关系数加2
                        self.tot_recall += 2
                    else: 
                        self.tot_recall +=  1

            sentence_boundaries = [0]
            words = []
            L = 0
            for i in range(len(sentences)):#一行数据中，即一个sentences中几个sentence
                L += len(sentences[i])#记录sentences有多长（多少个token）
                sentence_boundaries.append(L)
                words += sentences[i]

            tokens = [tokenize_word(w) for w in words]#将token分解成subword列表
            subwords = [w for li in tokens for w in li]
            maxL = max(maxL, len(subwords))
            subword2token = list(itertools.chain(*[[i] * len(li) for i, li in enumerate(tokens)]))
            token2subword = [0] + list(itertools.accumulate(len(li) for li in tokens))#每一个token在subword列表中的开始位置
            subword_start_positions = frozenset(token2subword)
            subword_sentence_boundaries = [sum(len(li) for li in tokens[:p]) for p in sentence_boundaries]
            #以上都和ner阶段的任务一样，以上的数据都是在一个sentences，即几个句子组成的整体中提取出的位置信息
            for n in range(len(subword_sentence_boundaries) - 1):#n个句子sentence
                #对每一个sentence作出处理，注意，虽然处理的单位是一个sentence，但是实际的输入依旧是整个sentences，只是有选择性的看
                sentence_ners = ners[n]
                sentence_relations = relations[n]
                std_ner = std_ners[n]

                std_entity_labels = {}
                self.ner_tot_recall += len(std_ner)#记录每个sentence有多少个ner关系

                for start, end, label in std_ner:
                    std_entity_labels[(start, end)] = label
                    self.ner_golden_labels.add( ((l_idx, n), (start, end), label) )#记录信息，文件第几行的第几个句子，实体边界是多少

                self.global_predicted_ners[(l_idx, n)] = list(sentence_ners)#global_predicted_ners[(l_idx, n)]里面存储了某一行某个句子所有的实体

                doc_sent_start, doc_sent_end = subword_sentence_boundaries[n : n + 2]#记录该句子sentence的边界


                #同样，在这个地方考虑语句的上下文信息
                left_length = doc_sent_start#左边的句子长度
                right_length = len(subwords) - doc_sent_end#右边的句子subword长度
                sentence_length = doc_sent_end - doc_sent_start
                half_context_length = int((max_num_subwords - sentence_length) / 2)

                if sentence_length < max_num_subwords:

                    if left_length < right_length:
                        left_context_length = min(left_length, half_context_length)
                        right_context_length = min(right_length, max_num_subwords - left_context_length - sentence_length)
                    else:
                        right_context_length = min(right_length, half_context_length)
                        left_context_length = min(left_length, max_num_subwords - right_context_length - sentence_length)


                doc_offset = doc_sent_start - left_context_length
                target_tokens = subwords[doc_offset : doc_sent_end + right_context_length]#几乎整个句子
                target_tokens = [tokenizer.cls_token] + target_tokens[ : self.max_seq_length - 4] + [tokenizer.sep_token] 
                assert(len(target_tokens) <= self.max_seq_length - 2)
                
                pos2label = {}
                for x in sentence_relations:
                    pos2label[(x[0],x[1],x[2],x[3])] = label_map[x[4]]
                    self.golden_labels.add(((l_idx, n), (x[0],x[1]), (x[2],x[3]), x[4]))#golden_labels_withner和golden_labels的区别在于加入了实体类型
                    self.golden_labels_withner.add(((l_idx, n), (x[0],x[1], std_entity_labels[(x[0], x[1])]), (x[2],x[3], std_entity_labels[(x[2], x[3])]), x[4]))
                    if x[4] in self.sym_labels[1:]:#如果是对称关系，则换个顺序在写一遍
                        self.golden_labels.add(((l_idx, n),  (x[2],x[3]), (x[0],x[1]), x[4]))
                        self.golden_labels_withner.add(((l_idx, n), (x[2],x[3], std_entity_labels[(x[2], x[3])]), (x[0],x[1], std_entity_labels[(x[0], x[1])]), x[4]))

                entities = list(sentence_ners)#某一个sentence中的实体

                for x in sentence_relations:#某一个sentence中的关系
                    w = (x[2],x[3],x[0],x[1])#反着来，即论文中所说的object到subject之间的关系
                    if w not in pos2label:
                        if x[4] in self.sym_labels[1:]:#如果是对称的关系，则加入pos2label中，因为在上述中，只是将对称关系的关系信息加入到了golden_labels中，但是并未加入到pos2label中
                            pos2label[w] = label_map[x[4]]  # bug
                        else:#如果不是对称关系，则将其加入到pos2label中，但是关系要变成另外一种格式，所以关系的类别共有8+8-3种
                            pos2label[w] = label_map[x[4]] + len(label_map) - len(self.sym_labels)

                if not self.evaluate:
                    entities.append((10000, 10000, 'NIL')) # only for NER
                #现在开始对一个实体对的subject进行操作
                for sub in entities: #sub代表一个实体信息（token信息）
                    cur_ins = []#每个subject都有一个单独的这个列表

                    if sub[0] < 10000:#如果实体的开始边界<10000
                        sub_s = token2subword[sub[0]] - doc_offset + 1#实体的subword的开始边界
                        sub_e = token2subword[sub[1]+1] - doc_offset#offest是subword和word的位置之差？
                        sub_label = ner_label_map[sub[2]]#标签信息

                        if self.use_typemarker:#此处用于定义是否将实体信息也加入到文本中
                            l_m = '[unused%d]' % ( 2 + sub_label )
                            r_m = '[unused%d]' % ( 2 + sub_label + len(self.ner_label_list))#根据之后的代码，可以判断，此处的2是说unused0和1已经被占用，剩下的，每ner_label_list长度为一个单位，顺序对应着subject和object的ner标签
                        else:#注：在vocab词汇表中，unused0的索引为1，以此类推
                            l_m = '[unused0]'
                            r_m = '[unused1]'
                        
                        sub_tokens = target_tokens[:sub_s] + [l_m] + target_tokens[sub_s:sub_e+1] + [r_m] + target_tokens[sub_e+1: ]#经过这一步，其实开始何结束位置指向的已经是这个标签的位置
                        sub_e += 2#在实体边界加上实体信息，因此边界值加2
                    else:
                        sub_s = len(target_tokens)
                        sub_e = len(target_tokens)+1
                        sub_tokens = target_tokens + ['[unused0]',  '[unused1]']
                        sub_label = -1

                    if sub_e >= self.max_seq_length-1:
                        continue
                    # assert(sub_e < self.max_seq_length)
                    for start, end, obj_label in sentence_ners:#在这一步开始对object实体进行操作，一个subject遍历所有的object
                        if self.model_type.endswith('nersub'):
                            if start==sub[0] and end==sub[1]:
                                continue

                        doc_entity_start = token2subword[start]
                        doc_entity_end = token2subword[end+1]
                        left = doc_entity_start - doc_offset + 1#object实体的subword边界
                        right = doc_entity_end - doc_offset

                        obj = (start, end)
                        if obj[0] >= sub[0]:
                            left += 1
                            if obj[0] > sub[1]:
                                left += 1

                        if obj[1] >= sub[0]:   
                            right += 1
                            if obj[1] > sub[1]:
                                right += 1#由最终的left和right，与object的边界值对比，就能判断出他们的相对位置关系
    
                        label = pos2label.get((sub[0], sub[1], obj[0], obj[1]), 0)#如果subject和当前object之间没关系，则会默认为0

                        if right >= self.max_seq_length-1:
                            continue

                        cur_ins.append(((left, right, ner_label_map[obj_label]), label, obj))#这个label指得是关系，obj是实体的位置信息，前者可能是object与subject的相对我位置关系
                        #cur_ins就是存储了当前subject和所有object之间的关系信息

                    maxR = max(maxR, len(cur_ins))
                    dL = self.max_pair_length#定义一个样本的最大关系对
                    if self.args.shuffle:
                        np.random.shuffle(cur_ins)

                    #包括这一步还是在一个subject的循环内做出的
                    for i in range(0, len(cur_ins), dL):#这个地方就是代表了一个样本的关系数。最大为16
                        examples = cur_ins[i : i + dL] #一个examples代表了一个subject和多个object之间的关系信息
                        item = {
                            'index': (l_idx, n),
                            'sentence': sub_tokens,
                            'examples': examples,
                            'sub': (sub, (sub_s, sub_e), sub_label), #(sub[0], sub[1], sub_label),
                        }                
                        #ner阶段的一个样本是一个sentence的所有实体对信息，但有最大限制256对，此处最大限制是16对关系数，并且句子的长度也限制到256
                        self.data.append(item) #也就是说，一个句子的一个subject和它所有的object组成一个样本，但object如果大于16，则会拆成两个及以上
        logger.info('maxR: %s', maxR)#关系的最大数
        logger.info('maxL: %s', maxL)#最长的sentences长度
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]#某一个样本
        sub, sub_position, sub_label = entry['sub']#sub指的是word 位置，不是subword位置
        input_ids = self.tokenizer.convert_tokens_to_ids(entry['sentence'])

        L = len(input_ids)
        input_ids += [self.tokenizer.pad_token_id] * (self.max_seq_length - len(input_ids))#后面用0补齐
        #max_entity_length 32，max pair 16  注意力矩阵
        attention_mask = torch.zeros((self.max_entity_length+self.max_seq_length, self.max_entity_length+self.max_seq_length), dtype=torch.int64)
        attention_mask[:L, :L] = 1#文本可以相互注意
        
        if self.model_type.startswith('albert'):#albert词汇表共有30002
            input_ids = input_ids + [30002] * (len(entry['examples'])) + [self.tokenizer.pad_token_id] * (self.max_pair_length - len(entry['examples']))
            input_ids = input_ids + [30003] * (len(entry['examples'])) + [self.tokenizer.pad_token_id] * (self.max_pair_length - len(entry['examples'])) # for debug
        else:
            input_ids = input_ids + [3] * (len(entry['examples'])) + [self.tokenizer.pad_token_id] * (self.max_pair_length - len(entry['examples']))
            input_ids = input_ids + [4] * (len(entry['examples'])) + [self.tokenizer.pad_token_id] * (self.max_pair_length - len(entry['examples'])) # for debug
            #3代表一个example的，即一个subject和object对的其中之一，4同理
        labels = []
        ner_labels = []
        mention_pos = []
        mention_2 = []
        position_ids = list(range(self.max_seq_length)) + [0] * self.max_entity_length #让256的文本内容具有position编码，后边的全是0
        num_pair = self.max_pair_length
        #examples代表的是object信息  例如((89, 90, 3), 0, (81, 82))，3指的是object的类别，一个example代表的是一个subject和其对应的所有object的关系列表，但是这个关系数不能太大
        for x_idx, obj in enumerate(entry['examples']):
            m2 = obj[0]
            label = obj[1]

            mention_pos.append((m2[0], m2[1]))#他们两个的相对位置信息，因为在任务中实体的相对关系也带着一定的含义
                                              #但是在实际的模型中并不一定使用
            mention_2.append(obj[2])#object的subword位置信息

            w1 = x_idx  
            w2 = w1 + num_pair

            w1 += self.max_seq_length
            w2 += self.max_seq_length
            #可见输入的语句是这样的  iiiiiiiiiioooooo ooooooo
            position_ids[w1] = m2[0]#object悬浮标记的开始标签共享开始subword的位置信息
            position_ids[w2] = m2[1]#同上

            for xx in [w1, w2]:#object的一对标签之间相互可见
                for yy in [w1, w2]:
                    attention_mask[xx, yy] = 1
                attention_mask[xx, :L] = 1#标签可见文本

            labels.append(label)#这个label指的是关系label
            ner_labels.append(m2[2])#object的label信息

            if self.use_typemarker:#如果使用了标签，则
                l_m = '[unused%d]' % ( 2 + m2[2] + len(self.ner_label_list)*2 )#m2[2]指代的是object的label信息
                r_m = '[unused%d]' % ( 2 + m2[2] + len(self.ner_label_list)*3 )
                l_m = self.tokenizer._convert_token_to_id(l_m)#把这个标签unusedx换成了对应的vocab表中的索引
                r_m = self.tokenizer._convert_token_to_id(r_m)
                input_ids[w1] = l_m#将object的类型标签的索引加入到相应的位置，而不再是unused0和unused1
                input_ids[w2] = r_m#同理


        pair_L = len(entry['examples'])
        if self.args.att_left:
            attention_mask[self.max_seq_length : self.max_seq_length+pair_L, self.max_seq_length : self.max_seq_length+pair_L] = 1
        if self.args.att_right:
            attention_mask[self.max_seq_length+num_pair : self.max_seq_length+num_pair+pair_L, self.max_seq_length+num_pair : self.max_seq_length+num_pair+pair_L] = 1

        mention_pos += [(0, 0)] * (num_pair - len(mention_pos))#不够16对的补齐16，每个位置代表一个对应位置的object的边界信息
        labels += [-1] * (num_pair - len(labels))#每个位置代表一个对应位置的object与suject的关系
        ner_labels += [-1] * (num_pair - len(ner_labels))#同上 补齐label列表（16）

        item = [torch.tensor(input_ids),
                attention_mask,
                torch.tensor(position_ids),
                torch.tensor(sub_position),
                torch.tensor(mention_pos),
                torch.tensor(labels, dtype=torch.int64),
                torch.tensor(ner_labels, dtype=torch.int64),
                torch.tensor(sub_label, dtype=torch.int64)
        ]
        #0代表inputid，1代表注意力矩阵，2代表position-id，3代表subject的位置信息，4代表两者的的位置信息，5代表关系类别，6代表object的类别，7代表subject的类别
        if self.evaluate:
            item.append(entry['index'])#8代表句子的索引#9代表subject的位置和类别#10代表所有object的位置信息
            item.append(sub)
            item.append(mention_2)

        return item

    @staticmethod
    def collate_fn(batch):
        fields = [x for x in zip(*batch)]

        num_metadata_fields = 3
        stacked_fields = [torch.stack(field) for field in fields[:-num_metadata_fields]]  # don't stack metadata fields
        stacked_fields.extend(fields[-num_metadata_fields:])  # add them as lists not torch tensors

        return stacked_fields

 


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)



def _rotate_checkpoints(args, checkpoint_prefix, use_mtime=False):
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    glob_checkpoints = glob.glob(os.path.join(args.output_dir, '{}-*'.format(checkpoint_prefix)))
    if len(glob_checkpoints) <= args.save_total_limit:
        return

    ordering_and_checkpoint_path = []
    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match('.*{}-([0-9]+)'.format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
        shutil.rmtree(checkpoint)

def train(args, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter("logs/"+args.data_dir[max(args.data_dir.rfind('/'),0):]+"_re_logs/"+args.output_dir[args.output_dir.rfind('/'):])

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    train_dataset = ACEDataset(tokenizer=tokenizer, args=args, max_pair_length=args.max_pair_length)

    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=4*int(args.output_dir.find('test')==-1))

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [#decay表示权重衰退，对参数正则化，用于防止过拟合，防止梯度爆炸
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]#两种优化器的参数
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    if args.warmup_steps==-1:#刚开始训练时,模型的权重(weights)是随机初始化的，此时若选择一个较大的学习率,可能带来模型的不稳定(振荡)，lr先慢慢增加，超过warmup_steps时，就采用初始设置的学习率
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(0.1*t_total), num_training_steps=t_total
        )
    else:#scheduler用于调节optimizer中的学习率
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
        )#因此，学习率确实是不断变化的，我们设置的只是一个初始学习率而已
    #初始学习率就是我们设置的学习率，由于预热的存在，scheduler会给优化器一个小的学习率，然后不断上升，到达预设值后根据学习率的更新算法调整优化器中的学习率
    if args.fp16:
        try:
            from apex import amp#混合精度运算
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
    # ori_model = model
    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    tr_ner_loss, logging_ner_loss = 0.0, 0.0
    tr_re_loss, logging_re_loss = 0.0, 0.0

    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    best_f1 = -1

    # 记录损失值用于后续的曲线绘制
    train_re_loss = []
    # 记录每次评估的f1指数
    train_re_f1 = []


    for _ in train_iterator:#开始
        if args.shuffle and _ > 0:
            train_dataset.initialize()
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            #一个batch为8个样本
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            # 0代表inputid，1代表注意力矩阵，2代表position-id，3代表subject的位置信息，4代表object的位置信息，5代表关系类别，6代表object的类别，7代表subject的类别
            inputs = {'input_ids':      batch[0],#288
                      'attention_mask': batch[1],#288*288
                      'position_ids':   batch[2],#288
                      'labels':         batch[5],#16，关系类别
                      'ner_labels':     batch[6],#16
                      }


            inputs['sub_positions'] = batch[3]
            if args.model_type.find('span')!=-1:
                inputs['mention_pos'] = batch[4]
            if args.model_type.endswith('bertonedropoutnersub'):
                inputs['sub_ner_labels'] = batch[7]

            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
            re_loss = outputs[1]#0代表loss，1代表re loss（subject和object的预测值相加作为预测得分计算），2代表ner loss（预测的是object的实体类型），3代表re预测得分，4代表ner预测得分
            ner_loss = outputs[2]

            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:# 如果梯度累积步骤大于1，则需要求平均损失，因为是累计步骤次才更新一次梯度
                loss = loss / args.gradient_accumulation_steps
                re_loss = re_loss / args.gradient_accumulation_steps
                ner_loss = ner_loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

             # 在此处保存re_loss值，用于绘制曲线
            train_re_loss.append(re_loss.item())

            tr_loss += loss.item()
            if re_loss > 0:
                tr_re_loss += re_loss.item()
            if ner_loss > 0:
                tr_ner_loss += ner_loss.item()


            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.max_grad_norm > 0:# 防止训练过程中梯度爆炸，进行梯度裁剪
                    if args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()#参数更新
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1#步骤数加一

                # if args.model_type.endswith('rel') :
                #     ori_model.bert.encoder.layer[args.add_coref_layer].attention.self.relative_attention_bias.weight.data[0].zero_() # 可以手动乘个mask

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
                    logging_loss = tr_loss

                    tb_writer.add_scalar('RE_loss', (tr_re_loss - logging_re_loss)/args.logging_steps, global_step)
                    logging_re_loss = tr_re_loss

                    tb_writer.add_scalar('NER_loss', (tr_ner_loss - logging_ner_loss)/args.logging_steps, global_step)
                    logging_ner_loss = tr_ner_loss

                #保存参数
                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0: # valid for bert/spanbert
                    update = True
                    # Save model checkpoint
                    if args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer)
                        f1 = results['f1_with_ner']
                        tb_writer.add_scalar('f1_with_ner', f1, global_step)

                        # 将f1指数保存到文件中，用于后续绘图,这个就是每次的评估阶段保存一次
                        train_re_f1.append(f1)

                        if f1 > best_f1:
                            best_f1 = f1
                            print ('Best F1', best_f1)
                        else:
                            update = False

                    if update:
                        checkpoint_prefix = 'checkpoint'
                        output_dir = os.path.join(args.output_dir, '{}-{}'.format(checkpoint_prefix, global_step))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training

                        model_to_save.save_pretrained(output_dir)

                        torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                        logger.info("Saving model checkpoint to %s", output_dir)

                        _rotate_checkpoints(args, checkpoint_prefix)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    # 保存损失值，用于后续的loss值绘制
    loss_output_dir = os.path.join(args.data_dir, data_dir['data_path_loss'])
    f1_output_dir = os.path.join(args.data_dir, data_dir['data_path_f1'])

    with open(loss_output_dir, 'w') as train_los:
        train_los.write(str(train_re_loss))

    with open(f1_output_dir, 'w') as train_f:
        train_f.write(str(train_re_f1))

    if args.local_rank in [-1, 0]:
        tb_writer.close()


    return global_step, tr_loss / global_step, best_f1

def to_list(tensor):
    return tensor.detach().cpu().tolist()

def evaluate(args, model, tokenizer, prefix="", do_test=False):

    eval_output_dir = args.output_dir

    eval_dataset = ACEDataset(tokenizer=tokenizer, args=args, evaluate=True, do_test=do_test, max_pair_length=args.max_pair_length)
    golden_labels = set(eval_dataset.golden_labels)
    golden_labels_withner = set(eval_dataset.golden_labels_withner)
    label_list = list(eval_dataset.label_list)
    sym_labels = list(eval_dataset.sym_labels)
    tot_recall = eval_dataset.tot_recall

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)


    scores = defaultdict(dict)
    # ner_pred = not args.model_type.endswith('noner')
    example_subs = set([])
    num_label = len(label_list)

    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()

    eval_sampler = SequentialSampler(eval_dataset) 
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,  collate_fn=ACEDataset.collate_fn, num_workers=4*int(args.output_dir.find('test')==-1))

    # Eval!
    logger.info("  Num examples = %d", len(eval_dataset))

    start_time = timeit.default_timer() 
    #8代表句子的索引#9代表subject的位置和类别#10代表所有object的位置信息
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        indexs = batch[-3]
        subs = batch[-2]
        batch_m2s = batch[-1]
        ner_labels = batch[6]

        batch = tuple(t.to(args.device) for t in batch[:-3])#去除后面三层

        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                    'attention_mask': batch[1],
                    'position_ids':   batch[2],
                    #   'labels':         batch[4],
                    #   'ner_labels':     batch[5],
                    }

            inputs['sub_positions'] = batch[3]
            if args.model_type.find('span')!=-1:
                inputs['mention_pos'] = batch[4]

            outputs = model(**inputs)

            logits = outputs[0]#re种类的预测得分16*16*13  样本数*实体对数*预测得分数
            #1代表object的ner得分
            if args.eval_logsoftmax:  # perform a bit better
                logits = torch.nn.functional.log_softmax(logits, dim=-1)#先softmax后取log

            elif args.eval_softmax:
                logits = torch.nn.functional.softmax(logits, dim=-1)

            #此处判断那是否要使用RE模型识别的实体类型
            if args.use_ner_results or args.model_type.endswith('nonersub'):#也就是说，这个地方用的是数据中正确的样本，而不是预测的值
                ner_preds = ner_labels#此处将ner结果记录下来
            else:
                ner_preds = torch.argmax(outputs[1], dim=-1)

            logits = logits.cpu().numpy()
            ner_preds = ner_preds.cpu().numpy()
            for i in range(len(indexs)):#多少个样本
                index = indexs[i]
                sub = subs[i]#因为每个样本就是一个subject和16个以内的object所组成的
                m2s = batch_m2s[i]#对应样本的所有object位置信息
                example_subs.add(((index[0], index[1]), (sub[0], sub[1])))
                for j in range(len(m2s)):#每个object
                    obj = m2s[j]
                    ner_label = eval_dataset.ner_label_list[ner_preds[i,j]]
                    scores[(index[0], index[1])][( (sub[0], sub[1]), (obj[0], obj[1]))] = (logits[i, j].tolist(), ner_label)#scores存储的就是每个subject和object对的关系
                    #此处的scores不再是一个样本16个固定大小了，他已经将所有的一句中的所有subject的所有object全部进入其中
    cor = 0 
    tot_pred = 0
    cor_with_ner = 0
    global_predicted_ners = eval_dataset.global_predicted_ners#这个是每个句子都有一个列表
    ner_golden_labels = eval_dataset.ner_golden_labels#这个是所有的实体都放在一个集合中
    ner_cor = 0 
    ner_tot_pred = 0
    ner_ori_cor = 0
    tot_output_results = defaultdict(list)
    #新增一个用来保存句子信息的文件
    sentence_output_results = defaultdict(list)

    if not args.eval_unidirect:     # eval_unidrect is for ablation study
        # print (len(scores))
        for example_index, pair_dict in sorted(scores.items(), key=lambda x:x[0]):#从所有的
            visited  = set([])
            sentence_results = []
            for k1, (v1, v2_ner_label) in pair_dict.items():#开始对一对实体进行处理
                #k1是subject和object的位置，v1是预测得分，v2是object的种类
                if k1 in visited:#看看当前的这对关系是否已经被处理过了
                    continue
                visited.add(k1)

                if v2_ner_label=='NIL':
                    continue
                v1 = list(v1)
                m1 = k1[0]
                m2 = k1[1]
                if m1 == m2:#如果subject和object一样，则跳过
                    continue
                k2 = (m2, m1)#反过来预测
                v2s = pair_dict.get(k2, None)#倒过来的关系预测得分
                if v2s is not None:
                    visited.add(k2)
                    v2, v1_ner_label = v2s#v2s有两个元素，分别如左边所说，v2是得分，v1_ner_label本来是object的种类，则对应于v1
                    v2 = v2[ : len(sym_labels)] + v2[num_label:] + v2[len(sym_labels) : num_label]#调整了一下顺序
                    #调整顺序的原因在于0-7是subject，8-12是object，颠倒顺序之后，将其也对换
                    for j in range(len(v2)):
                        v1[j] += v2[j]#正反顺序实体对的预测得分相加
                else:
                    assert ( False )

                if v1_ner_label=='NIL':
                    continue

                pred_label = np.argmax(v1)#得到预测的re关系
                if pred_label>0:
                    if pred_label >= num_label:#说明是颠倒的一种关系
                        pred_label = pred_label - num_label + len(sym_labels)#将其对应于0-7的位置
                        m1, m2 = m2, m1#颠倒subject和object，使其与上面的位置关系对应
                        v1_ner_label, v2_ner_label = v2_ner_label, v1_ner_label

                    pred_score = v1[pred_label]#将其映射到v1上

                    sentence_results.append( (pred_score, m1, m2, pred_label, v1_ner_label, v2_ner_label) )

            sentence_results.sort(key=lambda x: -x[0])#某一句话中的所有实体之间的预测关系
            no_overlap = []
            def is_overlap(m1, m2):
                if m2[0]<=m1[0] and m1[0]<=m2[1]:
                    return True
                if m1[0]<=m2[0] and m2[0]<=m1[1]:
                    return True
                return False

            output_preds = []
            output_preds_1 = []
            for item in sentence_results:
                m1 = item[1]
                m2 = item[2]
                overlap = False
                for x in no_overlap:
                    _m1 = x[1]
                    _m2 = x[2]
                    # same relation type & overlap subject & overlap object --> delete
                    if item[3]==x[3] and (is_overlap(m1, _m1) and is_overlap(m2, _m2)):
                        overlap = True
                        break

                pred_label = label_list[item[3]]


                if not overlap:
                    no_overlap.append(item)

            pos2ner = {}
            index = 0
            for item in no_overlap:
                m1 = item[1]
                m2 = item[2]
                pred_label = label_list[item[3]]
                tot_pred += 1#关系数加一
                if pred_label in sym_labels:
                    tot_pred += 1 # duplicate
                    if (example_index, m1, m2, pred_label) in golden_labels or (example_index, m2, m1, pred_label) in golden_labels:
                        cor += 2#对称关系则正确数加2
                else:
                    if (example_index, m1, m2, pred_label) in golden_labels:#判断预测的值和实际值之间的正确率
                        cor += 1        

                if m1 not in pos2ner:
                    pos2ner[m1] = item[4]
                if m2 not in pos2ner:
                    pos2ner[m2] = item[5]

                output_preds.append((m1, m2, pred_label))#加入到预测的关系列表中

                #新增列表来存储直观的数据
                file_path = os.path.join('sciner_models/sciner-scibert/ent_pred_test.json')
                f = open(file_path, "r", encoding='utf-8')
                num = 0
                data = []
                sentence = []
                sentences = []
                for l_idx, line in enumerate(f):  # 每次读一行sentences，包括很多个句子
                    if(num == example_index[0]):
                        data = json.loads(line)
                        break
                    num = num + 1

                sentences = data['sentences']
                length = 0
                for i in range(example_index[1]):
                    length =  length + len(sentences[i])
                sentence = sentences[example_index[1]]

                if index == 0:
                    output_preds_1.append(sentence)
                    index = index + 1

                output_preds_1.append((sentence[m1[0]-length:m1[1]+1-length], sentence[m2[0]-length:m2[1]+1-length], pred_label))

                if pred_label in sym_labels:
                    if (example_index, (m1[0], m1[1], pos2ner[m1]), (m2[0], m2[1], pos2ner[m2]), pred_label) in golden_labels_withner  \
                            or (example_index,  (m2[0], m2[1], pos2ner[m2]), (m1[0], m1[1], pos2ner[m1]), pred_label) in golden_labels_withner:
                        cor_with_ner += 2#预测中了实体类型
                else:  
                    if (example_index, (m1[0], m1[1], pos2ner[m1]), (m2[0], m2[1], pos2ner[m2]), pred_label) in golden_labels_withner:#判断是否也预测准了ner
                        cor_with_ner += 1      

            if do_test:#这一步将测试集的测试结果保存到一个列表中
                #output_w.write(json.dumps(output_preds) + '\n')
                tot_output_results[example_index[0]].append((example_index[1],  output_preds))

                sentence_output_results[example_index[0]].append((example_index[1],  output_preds_1))




            # refine NER results
            ner_results = list(global_predicted_ners[example_index])
            for i in range(len(ner_results)):
                start, end, label = ner_results[i] 
                if (example_index, (start, end), label) in ner_golden_labels:
                    ner_ori_cor += 1
                if (start, end) in pos2ner:
                    label = pos2ner[(start, end)]
                if (example_index, (start, end), label) in ner_golden_labels:
                    ner_cor += 1
                ner_tot_pred += 1
        
    else:

        for example_index, pair_dict in sorted(scores.items(), key=lambda x:x[0]):  
            sentence_results = []
            for k1, (v1, v2_ner_label) in pair_dict.items():
                
                if v2_ner_label=='NIL':
                    continue
                v1 = list(v1)
                m1 = k1[0]
                m2 = k1[1]
                if m1 == m2:
                    continue
              
                pred_label = np.argmax(v1)
                if pred_label>0 and pred_label < num_label:

                    pred_score = v1[pred_label]

                    sentence_results.append( (pred_score, m1, m2, pred_label, None, v2_ner_label) )

            sentence_results.sort(key=lambda x: -x[0])
            no_overlap = []
            def is_overlap(m1, m2):
                if m2[0]<=m1[0] and m1[0]<=m2[1]:
                    return True
                if m1[0]<=m2[0] and m2[0]<=m1[1]:
                    return True
                return False

            output_preds = []

            for item in sentence_results:
                m1 = item[1]
                m2 = item[2]
                overlap = False
                for x in no_overlap:
                    _m1 = x[1]
                    _m2 = x[2]
                    if item[3]==x[3] and (is_overlap(m1, _m1) and is_overlap(m2, _m2)):
                        overlap = True
                        break

                pred_label = label_list[item[3]]

                output_preds.append((m1, m2, pred_label))

                if not overlap:
                    no_overlap.append(item)

            pos2ner = {}
            predpos2ner = {}
            ner_results = list(global_predicted_ners[example_index])
            for start, end, label in ner_results:
                predpos2ner[(start, end)] = label

            for item in no_overlap:
                m1 = item[1]
                m2 = item[2]
                pred_label = label_list[item[3]]
                tot_pred += 1

                if (example_index, m1, m2, pred_label) in golden_labels:
                    cor += 1        

                if m1 not in pos2ner:
                    pos2ner[m1] = predpos2ner[m1]#item[4]

                if m2 not in pos2ner:
                    pos2ner[m2] = item[5]

                # if pred_label in sym_labels:
                #     if (example_index, (m1[0], m1[1], pos2ner[m1]), (m2[0], m2[1], pos2ner[m2]), pred_label) in golden_labels_withner \
                #         or (example_index,  (m2[0], m2[1], pos2ner[m2]), (m1[0], m1[1], pos2ner[m1]), pred_label) in golden_labels_withner:
                #         cor_with_ner += 2
                # else:  #判断是否连ner类型也判断正确
                if (example_index, (m1[0], m1[1], pos2ner[m1]), (m2[0], m2[1], pos2ner[m2]), pred_label) in golden_labels_withner:
                    cor_with_ner += 1      
            
            # refine NER results
            ner_results = list(global_predicted_ners[example_index])
            for i in range(len(ner_results)):
                start, end, label = ner_results[i] 
                if (example_index, (start, end), label) in ner_golden_labels:
                    ner_ori_cor += 1
                if (start, end) in pos2ner:
                    label = pos2ner[(start, end)]
                if (example_index, (start, end), label) in ner_golden_labels:
                    ner_cor += 1
                ner_tot_pred += 1


    evalTime = timeit.default_timer() - start_time
    logger.info("  Evaluation done in total %f secs (%f example per second)", evalTime,  len(global_predicted_ners) / evalTime)

    if do_test:
        output_w = open(os.path.join(args.output_dir, 'pred_results.json'), 'w')
        json.dump(tot_output_results, output_w)
    #新增保存具体实体的文件
        output_w_1 = open(os.path.join(args.output_dir, 'pred_results_sentence.json'), 'w')
        json.dump(sentence_output_results, output_w_1)

    ner_p = ner_cor / ner_tot_pred if ner_tot_pred > 0 else 0 
    ner_r = ner_cor / len(ner_golden_labels) 
    ner_f1 = 2 * (ner_p * ner_r) / (ner_p + ner_r) if ner_cor > 0 else 0.0

    p = cor / tot_pred if tot_pred > 0 else 0 
    r = cor / tot_recall 
    f1 = 2 * (p * r) / (p + r) if cor > 0 else 0.0
    assert(tot_recall==len(golden_labels))

    p_with_ner = cor_with_ner / tot_pred if tot_pred > 0 else 0 
    r_with_ner = cor_with_ner / tot_recall
    assert(tot_recall==len(golden_labels_withner))
    f1_with_ner = 2 * (p_with_ner * r_with_ner) / (p_with_ner + r_with_ner) if cor_with_ner > 0 else 0.0

    results = {'f1':  f1,  'f1_with_ner': f1_with_ner, 'ner_f1': ner_f1}

    logger.info("Result: %s", json.dumps(results))

    return results

# 读取存储为txt文件的数据
def data_read(dir_path):
    with open(dir_path, "r") as f:
        raw_data = f.read()
        data = raw_data[1:-1].split(", ")# [-1:1]是为了去除文件中的前后中括号"[]"

    return np.asfarray(data, float)

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default='ace_data', type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=384, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")

    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=2e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=10.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=-1, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=5,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=1000,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    parser.add_argument('--save_total_limit', type=int, default=1,
                        help='Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default')

    parser.add_argument("--train_file",  default="train.json", type=str)
    parser.add_argument("--dev_file",  default="dev.json", type=str)
    parser.add_argument("--test_file",  default="test.json", type=str)
    parser.add_argument('--max_pair_length', type=int, default=64,  help="")
    parser.add_argument("--alpha", default=1.0, type=float)
    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--no_test', action='store_true')
    parser.add_argument('--eval_logsoftmax', action='store_true')
    parser.add_argument('--eval_softmax', action='store_true')
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--lminit', action='store_true')
    parser.add_argument('--no_sym', action='store_true')
    parser.add_argument('--att_left', action='store_true')
    parser.add_argument('--att_right', action='store_true')
    parser.add_argument('--use_ner_results', action='store_true')
    parser.add_argument('--use_typemarker', action='store_true')
    parser.add_argument('--eval_unidirect', action='store_true')

    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    def create_exp_dir(path, scripts_to_save=None):
        if args.output_dir.endswith("test"):
            return
        if not os.path.exists(path):
            os.mkdir(path)

        print('Experiment dir : {}'.format(path))
        if scripts_to_save is not None:
            if not os.path.exists(os.path.join(path, 'scripts')):
                os.mkdir(os.path.join(path, 'scripts'))
            for script in scripts_to_save:
                dst_file = os.path.join(path, 'scripts', os.path.basename(script))
                shutil.copyfile(script, dst_file)

    if args.do_train and args.local_rank in [-1, 0] and args.output_dir.find('test')==-1:
        create_exp_dir(args.output_dir, scripts_to_save=['run_re.py', 'transformers/src/transformers/modeling_bert.py', 'transformers/src/transformers/modeling_albert.py'])


    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:#local_rank == -1 表示不适用分布式训练
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    if args.data_dir.find('ace')!=-1:
        num_ner_labels = 8

        if args.no_sym:
            num_labels = 7 + 7 - 1
        else:
            num_labels = 7 + 7 - 2
    elif args.data_dir.find('scierc')!=-1:
        num_ner_labels = 7

        if args.no_sym:
            num_labels = 8 + 8 - 1
        else:
            num_labels = 8 + 8 - 3#为什么？
    else:
        assert (False)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab


    args.model_type = args.model_type.lower()#model_type == bertsub

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]#model_class = BertForACEBothOneDropoutSub

    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path, num_labels=num_labels)
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path,  do_lower_case=args.do_lower_case)

    config.max_seq_length = args.max_seq_length
    config.alpha = args.alpha
    config.num_ner_labels = num_ner_labels

    model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config)


    if args.model_type.startswith('albert'):
        if args.use_typemarker:
            special_tokens_dict = {'additional_special_tokens': ['[unused' + str(x) + ']' for x in range(num_ner_labels*4+2)]}
        else:
            special_tokens_dict = {'additional_special_tokens': ['[unused' + str(x) + ']' for x in range(4)]}
        tokenizer.add_special_tokens(special_tokens_dict)
        # print ('add tokens:', tokenizer.additional_special_tokens)
        # print ('add ids:', tokenizer.additional_special_tokens_ids)
        model.albert.resize_token_embeddings(len(tokenizer))

    if args.do_train:
        subject_id = tokenizer.encode('subject', add_special_tokens=False)#得到subject这个单词的token id列表
        assert(len(subject_id)==1)
        subject_id = subject_id[0]#得到具体的数值
        object_id = tokenizer.encode('object', add_special_tokens=False)
        assert(len(object_id)==1)
        object_id = object_id[0]#与上述一致

        mask_id = tokenizer.encode('[MASK]', add_special_tokens=False)#同理
        if args.model_type.startswith('bert'):#在robert中会有所不同，robert的encode返回值有三个
            assert (len(mask_id) == 1)
        mask_id = mask_id[0]

        logger.info(" subject_id = %s, object_id = %s, mask_id = %s", subject_id, object_id, mask_id)

        if args.lminit: 
            if args.model_type.startswith('albert'):
                word_embeddings = model.albert.embeddings.word_embeddings.weight.data
                subs = 30000
                sube = 30001
                objs = 30002
                obje = 30003
            else:
                word_embeddings = model.bert.embeddings.word_embeddings.weight.data
                subs = 1
                sube = 2
                objs = 3
                obje = 4

            word_embeddings[subs].copy_(word_embeddings[mask_id])     
            word_embeddings[sube].copy_(word_embeddings[subject_id])   

            word_embeddings[objs].copy_(word_embeddings[mask_id])      
            word_embeddings[obje].copy_(word_embeddings[object_id])     

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)
    best_f1 = 0
    # Training
    if args.do_train:
        # train_dataset = load_and_cache_examples(args,  tokenizer, evaluate=False)
        global_step, tr_loss, best_f1 = train(args, model, tokenizer)
        # 绘制loss曲线
        train_loss_path = os.path.join(args.data_dir, data_dir['data_path_loss'])  # 存储文件路径
        y_train_loss = data_read(train_loss_path)  # loss值，即y轴
        x_train_loss = range(len(y_train_loss))  # loss的数量，即x轴

        plt.figure()

        # 去除顶部和右边框框
        ax = plt.axes()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # plt.ylim(0, 0.5)
        plt.xlabel('iters')  # x轴标签
        plt.ylabel('loss')  # y轴标签

        # 以x_train_loss为横坐标，y_train_loss为纵坐标，曲线宽度为1，实线，增加标签，训练损失，
        # 默认颜色，如果想更改颜色，可以增加参数color='red',这是红色。
        plt.plot(x_train_loss, y_train_loss, linewidth=1, linestyle="solid", label="train_loss")
        plt.legend()
        plt.title(data_dir['data_name_loss'] + 'curve')
        plt.show()

        # 绘制f1指数曲线
        train_f1_path = os.path.join(args.data_dir, data_dir['data_path_f1'])  # 存储文件路径# 存储文件路径

        y_train_acc = data_read(train_f1_path)  # 训练准确率值，即y轴
        x_train_acc = range(len(y_train_acc))  # 训练阶段准确率的数量，即x轴

        plt.figure()

        # 去除顶部和右边框框
        ax = plt.axes()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.xlabel('epochs')  # x轴标签
        plt.ylabel('re_f1')  # y轴标签

        # 以x_train_acc为横坐标，y_train_acc为纵坐标，曲线宽度为1，实线，增加标签，训练损失，
        # 增加参数color='red',这是红色。
        plt.plot(x_train_acc, y_train_acc, color='red', linewidth=1, linestyle="solid", label="train_f1")
        plt.legend()
        plt.title(data_dir['data_name_f1'] + 'curve')
        plt.show()


        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)


    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)
        update = True
        if args.evaluate_during_training:
            results = evaluate(args, model, tokenizer)
            f1 = results['f1_with_ner']
            if f1 > best_f1:
                best_f1 = f1
                print ('Best F1', best_f1)
            else:
                update = False

        if update:
            checkpoint_prefix = 'checkpoint'
            output_dir = os.path.join(args.output_dir, '{}-{}'.format(checkpoint_prefix, global_step))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training

            model_to_save.save_pretrained(output_dir)

            torch.save(args, os.path.join(output_dir, 'training_args.bin'))
            logger.info("Saving model checkpoint to %s", output_dir)
            _rotate_checkpoints(args, checkpoint_prefix)

        tokenizer.save_pretrained(args.output_dir)

        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))


    # Evaluation
    results = {'dev_best_f1': best_f1}
    if args.do_eval and args.local_rank in [-1, 0]:

        checkpoints = [args.output_dir]

        WEIGHTS_NAME = 'pytorch_model.bin'

        if args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))

        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        #包括NER结果的测试
        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""

            model = model_class.from_pretrained(checkpoint, config=config)

            model.to(args.device)
            result = evaluate(args, model, tokenizer, prefix=global_step, do_test=not args.no_test)


            result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
            results.update(result)
        print (results)

        if args.no_test:  # choose best resutls on dev set
            bestv = 0
            k = 0
            for k, v in results.items():
                if v > bestv:
                    bestk = k
            print (bestk)

        output_eval_file = os.path.join(args.output_dir, "results.json")
        json.dump(results, open(output_eval_file, "w"))


if __name__ == "__main__":
    main()


