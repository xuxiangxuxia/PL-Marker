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

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random
from collections import defaultdict
import re
import shutil
import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange

from transformers import (WEIGHTS_NAME, BertConfig,
                          BertTokenizer,
                          RobertaConfig,
                          RobertaTokenizer,
                          get_linear_schedule_with_warmup,
                          AdamW,
                          BertForMaskedLM,
                          BertForNER,
                          BertForSpanNER,
                          BertForSpanMarkerNER,
                          BertForSpanMarkerNER_onlymarker,
                          BertForSpanMarkerNER_onlyentity,
                          BertForSpanMarkerBiNER,
                          AlbertForNER,
                          AlbertConfig,
                          AlbertTokenizer,
                          BertForLeftLMNER,
                          RobertaForNER,
                          RobertaForSpanNER,
                          RobertaForSpanMarkerNER,
                          AlbertForSpanNER,
                          AlbertForSpanMarkerNER,
                          )

from transformers import AutoTokenizer
from torch.utils.data import TensorDataset, Dataset
import json
import pickle
import numpy as np
import unicodedata
import itertools
import math
from tqdm import tqdm
import re
import timeit

logger = logging.getLogger(__name__)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, RobertaConfig)), ())

MODEL_CLASSES = {
    'bert': (BertConfig, BertForNER, BertTokenizer),
    #'bert_prompt': (BertConfig, BertForSpanMarkerNER_prompt, BertTokenizer),
    'bertspan': (BertConfig, BertForSpanNER, BertTokenizer),
    'bertspanmarker': (BertConfig, BertForSpanMarkerNER, BertTokenizer),
    'bertspanmarker_onlymarker': (BertConfig, BertForSpanMarkerNER_onlymarker, BertTokenizer),
    'bertspanmarker_onlyentity': (BertConfig, BertForSpanMarkerNER_onlyentity, BertTokenizer),
    'bertspanmarkerbi': (BertConfig, BertForSpanMarkerBiNER, BertTokenizer),
    'bertleftlm': (BertConfig, BertForLeftLMNER, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForNER, RobertaTokenizer),
    'robertaspan': (RobertaConfig, RobertaForSpanNER, RobertaTokenizer),
    'robertaspanmarker': (RobertaConfig, RobertaForSpanMarkerNER, RobertaTokenizer),
    'albert': (AlbertConfig, AlbertForNER, AlbertTokenizer),
    'albertspan': (AlbertConfig, AlbertForSpanNER, AlbertTokenizer),
    'albertspanmarker': (AlbertConfig, AlbertForSpanMarkerNER, AlbertTokenizer),
}

# data_dir={'data_path_loss':'train_ner_loss.txt','data_name_loss':'train_ner_loss','data_path_f1':'train_ner_f1.txt','data_name_f1':'train_ner_f1'}
# data_dir={'data_path_loss':'train_ner_onlyentity_loss.txt','data_name_loss':'train_ner_onlyentity_loss','data_path_f1':'train_ner_onlyentity_f1.txt','data_name_f1':'train_ner_onlyentity_f1'}
data_dir = {'data_path_loss': 'train_ner_prompt_loss.txt', 'data_name_loss': 'train_ner_prompt_loss',
            'data_path_f1': 'train_ner_prompt_f1.txt', 'data_name_f1': 'train_ner_prompt_f1'}


class ACEDatasetNER(Dataset):
    def __init__(self, tokenizer, args=None, evaluate=False, do_test=False):
        if not evaluate:
            file_path = os.path.join(args.data_dir, args.train_file)
        else:
            if do_test:
                file_path = os.path.join(args.data_dir, args.test_file)
            else:
                file_path = os.path.join(args.data_dir, args.dev_file)

        assert os.path.isfile(file_path)

        self.file_path = file_path

        self.tokenizer = tokenizer
        self.max_seq_length = args.max_seq_length

        self.evaluate = evaluate
        self.local_rank = args.local_rank
        self.args = args
        self.model_type = args.model_type

        if args.data_dir.find('ace') != -1:
            self.ner_label_list = ['NIL', 'FAC', 'WEA', 'LOC', 'VEH', 'GPE', 'ORG', 'PER']
        elif args.data_dir.find('scierc') != -1:
            self.ner_label_list = ['NIL', 'Method', 'OtherScientificTerm', 'Task', 'Generic', 'Material', 'Metric']
        else:
            self.ner_label_list = ['NIL', 'CARDINAL', 'DATE', 'EVENT', 'FAC', 'GPE', 'LANGUAGE', 'LAW', 'LOC', 'MONEY',
                                   'NORP', 'ORDINAL', 'ORG', 'PERCENT', 'PERSON', 'PRODUCT', 'QUANTITY', 'TIME',
                                   'WORK_OF_ART']

        self.max_pair_length = args.max_pair_length
        # 最多多少个实体？512为最大的token数，也为最大的实体数
        self.max_entity_length = args.max_pair_length * 2
        self.initialize()  # 对数据集初始化

    def is_punctuation(self, char):
        # obtained from:
        # https://github.com/huggingface/transformers/blob/5f25a5f367497278bf19c9994569db43f96d5278/transformers/tokenization_bert.py#L489
        cp = ord(char)
        if (cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126):
            return True
        cat = unicodedata.category(char)
        if cat.startswith("P"):
            return True
        return False

    def get_original_token(self, token):
        escape_to_original = {
            "-LRB-": "(",
            "-RRB-": ")",
            "-LSB-": "[",
            "-RSB-": "]",
            "-LCB-": "{",
            "-RCB-": "}",
        }
        if token in escape_to_original:
            token = escape_to_original[token]
        return token

    def initialize(self):
        tokenizer = self.tokenizer
        # 510
        max_num_subwords = self.max_seq_length - 2

        ner_label_map = {label: i for i, label in enumerate(self.ner_label_list)}  # 实体类型的对应索引数字

        def tokenize_word(text):
            if (
                    isinstance(tokenizer, RobertaTokenizer)
                    and (text[0] != "'")
                    and (len(text) != 1 or not self.is_punctuation(text))
            ):
                return tokenizer.tokenize(text, add_prefix_space=True)  # 指定token为可分割的整体(即面向subword)
            return tokenizer.tokenize(text)

        f = open(self.file_path, "r", encoding='utf-8')
        self.data = []
        self.tot_recall = 0
        self.ner_golden_labels = set([])
        maxL = 0
        maxR = 0

        for l_idx, line in enumerate(f):
            data = json.loads(line)
            # if len(self.data) > 5:
            #     break
            if self.args.output_dir.find('test') != -1:
                if len(self.data) > 5:
                    break
            sentences = data['sentences']
            for i in range(len(sentences)):
                for j in range(len(sentences[i])):
                    sentences[i][j] = self.get_original_token(sentences[i][j])  # 将括号用特殊符号代替

            ners = data['ner']

            sentence_boundaries = [0]
            words = []
            L = 0
            for i in range(len(sentences)):
                L += len(sentences[i])  # 共有多少个单词
                sentence_boundaries.append(L)  # boundaries是一个[0,22,60]类型的分隔符号,每个符号代表一个句子的结束处,22实际上指的是下一个句子的开始
                words += sentences[i]  # 每个单词组成列表
            # 拆分成subword
            tokens = [tokenize_word(w) for w in words]
            subwords = [w for li in tokens for w in li]  # li中是一个单词分解成的subword序列，里面有很多subword
            maxL = max(len(tokens), maxL)  # tokens的长度
            subword2token = list(itertools.chain(*[[i] * len(li) for i, li in enumerate(tokens)]))  # i是token的索引,把subword映射到对应的token
            token2subword = [0] + list(itertools.accumulate(len(li) for li in tokens))  # 每个token的第一个subword的序号
            subword_start_positions = frozenset(token2subword)
            subword_sentence_boundaries = [sum(len(li) for li in tokens[:p]) for p in sentence_boundaries]  # senten的subword边界，

            for n in range(len(subword_sentence_boundaries) - 1):
                sentence_ners = ners[n]  #每个句子的ner序列

                self.tot_recall += len(sentence_ners)
                entity_labels = {}
                for start, end, label in sentence_ners:
                    entity_labels[(token2subword[start], token2subword[end + 1])] = ner_label_map[label]  # {(0, 1): 5}这样的形式，左闭右开，代表一个实体在subword列表中的边界
                    self.ner_golden_labels.add(((l_idx, n), (start, end),label))  # l_index代表是文件中的第几行，n代表这一行中的第几个句子，ner_golden_labels代表整个文件中的实体

                doc_sent_start, doc_sent_end = subword_sentence_boundaries[n: n + 2]  # 从n到n+2,[n,n+2)的边界，即，n和n+1的值，一次一个句子

                left_length = doc_sent_start  # 当前句子之前的长度
                right_length = len(subwords) - doc_sent_end  # 剩余的句子的subword长度
                sentence_length = doc_sent_end - doc_sent_start  # 句子的长度
                half_context_length = int((max_num_subwords - sentence_length) / 2)  # （510-句子长度）/2

                if left_length < right_length:  # 判断左右句子的长度
                    left_context_length = min(left_length,half_context_length)  # 这个可能是必然会实现left_context_length = left_length，因为left_length + right_length = 2*half_context_length
                    right_context_length = min(right_length,max_num_subwords - left_context_length - sentence_length)  # 好像必然是right_context_length = right_length
                else:
                    right_context_length = min(right_length, half_context_length)
                    left_context_length = min(left_length, max_num_subwords - right_context_length - sentence_length)
                if self.args.output_dir.find('ctx0') != -1:
                    left_context_elngth = right_context_length = 0  # for debug

                doc_offset = doc_sent_start - left_context_length  # 偏移度，句子的开始减去左边文章的开始
                target_tokens = subwords[doc_offset: doc_sent_end + right_context_length]  # 之所以这样做是为了截取当前句子的上下文语境
                assert (len(target_tokens) <= max_num_subwords)

                # 修改，此处需要将prompt句子也加入其中
                target_tokens = [tokenizer.cls_token] + target_tokens +["[UNK]","[UNK]","is","a","[MASK]","entity","."]+ [tokenizer.sep_token]  # 这才是输入到bert中的一个句子的标准形式

                #target_tokens = [tokenizer.cls_token] + target_tokens +  [tokenizer.sep_token]



                entity_infos = []
                # 两个循环，便利所有可能的subword位置组合，找到实体subword对


                for entity_start in range(left_context_length, left_context_length + sentence_length ):
                    doc_entity_start = entity_start + doc_offset  # doc_offset = doc_sent_start - left_context_length，即每个句子的开始subword
                    if doc_entity_start not in subword_start_positions:  # 同理，判断是否是一个token的开始，是则继续，不是则换下一个
                        continue
                    for entity_end in range(entity_start + 1,left_context_length + sentence_length + 1):  # 其实就是遍历这个句子的长度
                        doc_entity_end = entity_end + doc_offset  # doc_entity_end其实就是subword的序号，代表每一次循环的边界subword
                        if doc_entity_end not in subword_start_positions:  # 如果当前的doc_entity_end不是一个token的开始subword，则会进入，因为考虑实体只对token层次看，而如果是一个token里面的，则没必要接着看每一个subword
                            continue

                        if subword2token[doc_entity_end - 1] - subword2token[
                            doc_entity_start] + 1 > self.args.max_mention_ori_length:  # 一旦两个词之间的距离过大，则会直接返回
                            continue

                        label = entity_labels.get((doc_entity_start, doc_entity_end), 0)
                        entity_labels.pop((doc_entity_start, doc_entity_end),None)  # 将上述的label成员从字典中移除,当然，默认为0的不在label里，故删不了什么东西
                        entity_infos.append(((entity_start + 1, entity_end), label, (subword2token[doc_entity_start],subword2token[doc_entity_end - 1])))  # ((1, 1), 5, (0, 0))，其中前面的是subword边界，后面的是实体类型，默认为0，后面为token的边界
                # if len(entity_labels):
                #     print ((entity_labels))
                # assert(len(entity_labels)==0)

                # dL = self.max_pair_length
                # maxR = max(maxR, len(entity_infos))
                # for i in range(0, len(entity_infos), dL):
                #     examples = entity_infos[i : i + dL]
                #     item = {
                #         'sentence': target_tokens,
                #         'examples': examples,
                #         'example_index': (l_idx, n),
                #         'example_L': len(entity_infos)
                #     }

                #     self.data.append(item)
                maxR = max(maxR, len(entity_infos))  # 找到这种实体关系的最大数
                dL = self.max_pair_length
                if self.args.shuffle:
                    random.shuffle(entity_infos)
                if self.args.group_sort:
                    group_axis = np.random.randint(2)
                    sort_dir = bool(np.random.randint(2))
                    entity_infos.sort(key=lambda x: (x[0][group_axis], x[0][1 - group_axis]), reverse=sort_dir)

                if not self.args.group_edge:
                    for i in range(0, len(entity_infos), dL):  # 从0-len，其中dl为步进值

                        examples = entity_infos[i: i + dL]  # 就是将实体的信息放入examples中，用于后续的判断，每一个example的长度为dl
                        item = {  # 这个字典记录了每个句子中的实体关系信息，以及句子信息
                            'sentence': target_tokens,
                            'examples': examples,
                            'example_index': (l_idx, n),  # 同样的，第几行第几个句子
                            'example_L': len(entity_infos)
                        }  # 此处的item是送往dataset中的句子，即送入bert的句子，example相当于一中验证，用于和bert输出的识别结果做比对
                        self.data.append(item)
                else:
                    if self.args.group_axis == -1:
                        group_axis = np.random.randint(2)
                    else:
                        group_axis = self.args.group_axis
                    sort_dir = bool(np.random.randint(2))
                    entity_infos.sort(key=lambda x: (x[0][group_axis], x[0][1 - group_axis]), reverse=sort_dir)
                    _start = 0
                    while _start < len(entity_infos):
                        _end = _start + dL
                        if _end >= len(entity_infos):
                            _end = len(entity_infos)
                        else:
                            while entity_infos[_end - 1][0][group_axis] == entity_infos[_end][0][
                                group_axis] and _end > _start:
                                _end -= 1
                            if _start == _end:
                                _end = _start + dL

                        examples = entity_infos[_start: _end]

                        item = {
                            'sentence': target_tokens,
                            'examples': examples,
                            'example_index': (l_idx, n),
                            'example_L': len(entity_infos)
                        }

                        self.data.append(item)
                        _start = _end

        logger.info('maxL: %d', maxL)  # 每一个句子token的最大数
        logger.info('maxR: %d', maxR)  # 每个句子中的entity_infos的最大值（entity_infos就是任意两个位置之间的subword的合计）

        # exit()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]

        input_ids = self.tokenizer.convert_tokens_to_ids(entry['sentence'])
        L = len(input_ids)  # 相当于句子的subword长度

        input_ids += [0] * (self.max_seq_length - len(input_ids)) # 用0来把后面的补全，这就是传统attention-mask的任务，使得bert在子注意力时选择性丢掉不必要的信息
        position_plus_pad = int(self.model_type.find('roberta') != -1) * 2

        if self.model_type not in ['bertspan', 'robertaspan', 'albertspan']:

            if self.model_type.startswith('albert'):
                input_ids = input_ids + [30000] * (len(entry['examples'])) + [0] * (self.max_pair_length - len(entry['examples']))
                input_ids = input_ids + [30001] * (len(entry['examples'])) + [0] * (self.max_pair_length - len(entry['examples']))
            elif self.model_type.startswith('roberta'):
                input_ids = input_ids + [50261] * (len(entry['examples'])) + [0] * (self.max_pair_length - len(entry['examples']))
                input_ids = input_ids + [50262] * (len(entry['examples'])) + [0] * (self.max_pair_length - len(entry['examples']))
            else:  # bertspanmarker    在原来句子后面加上标记实体关系的悬浮标记，max_pair_length代表最多有多少个实体，
                #input_ids = input_ids + [1] * (len(entry['examples'])) + [0] * (self.max_pair_length - len(entry['examples']))
                #input_ids = input_ids + [2] * (len(entry['examples'])) + [0] * (self.max_pair_length - len(entry['examples']))

                #修改每个example的实体对为1后，就不需要在将后面的补全了，也就是说，以后，input_ids的值长度为514
                input_ids = input_ids + [1] * (len(entry['examples']))
                input_ids = input_ids + [2] * (len(entry['examples']))

            attention_mask = torch.zeros((self.max_entity_length + self.max_seq_length, self.max_entity_length + self.max_seq_length),dtype=torch.int64)
            attention_mask[:L, :L] = 1  # 就是定向注意力矩阵，此处让文本信息可以相互可见
            position_ids = list(range(position_plus_pad, position_plus_pad + self.max_seq_length)) + [0] * self.max_entity_length

        else:
            attention_mask = [1] * L + [0] * (self.max_seq_length - L)
            attention_mask = torch.tensor(attention_mask, dtype=torch.int64)
            position_ids = list(range(position_plus_pad, position_plus_pad + self.max_seq_length)) + [0] * self.max_entity_length
            # position_ids 句子的posionid从0-511，后面的（实体信息）全部为0.

        labels = []
        mentions = []
        mention_pos = []
        num_pair = self.max_pair_length
        mask_id = []

        full_attention_mask = [1] * L + [0] * (self.max_seq_length - L) + [0] * (self.max_pair_length) * 2
        # 全注意力矩阵用于使得句子之间相互可见
        for x_idx, x in enumerate(entry['examples']):#此时只有1对关系
            m1 = x[0]
            label = x[1]
            mentions.append(x[2])  # 实体的word边界信息# ((1, 1), 5, (0, 0))，其中前面的是subword边界，后面的是实体类型，默认为0，后面为token的边界
            mention_pos.append((m1[0], m1[1])) #实体的subword边界的位置信息
            labels.append(label)

            if self.model_type in ['bertspan', 'robertaspan', 'albertspan']:
                continue

            w1 = x_idx
            w2 = w1 + num_pair  # 这个其实就是两个实体的边界信息相隔多远，距离其实是个固定的数（256），句子长度为512，后面的256为开始边界，在后面的256为结束边界。因为在本代码中，两个实体的边界信息是分开存储的，一前一后
            # 这个地方也就可以理解为什么一个example最大有256个实体信息，因为在此处只能有（1024-512）/2=256对实体边界信息
            w1 += self.max_seq_length
            w2 += self.max_seq_length
            position_ids[w1] = m1[0]  # 共享位置信息，句子后方的，实体的开始边界共享剧中的信息
            position_ids[w2] = m1[1]  # 同理，实体的结束边界也共享句中的位置信息

            #此处修改，将prompt中的UNK也与该实体的位置信息共享
            position_ids[L-8] = m1[0] #l为input的长度
            position_ids[L-7] = m1[1]

            input_ids[L-8] = input_ids[m1[0]]#也把边界信息共享
            input_ids[L-7] = input_ids[m1[1]]


            for xx in [w1, w2]:
                full_attention_mask[xx] = 1
                for yy in [w1, w2]:  # yy取值就是这两个数
                    attention_mask[xx, yy] = 1  # attention mask就是定向注意力矩阵，这两个marker之间是可以相互看到的，为1
                attention_mask[xx, :L] = 1  # 该标记项也可以看到所有的文本数据

        labels += [-1] * (num_pair - len(labels))  # 同理，补全至2
        mention_pos += [(0, 0)] * (num_pair - len(mention_pos))  # 用（0，0）的位置信息来补全1个，实际就是不补全
        mask_id.append(input_ids.index(self.tokenizer.mask_token_id))

        item = [torch.tensor(input_ids),  # 最终的inputid就是包含句子(512)和实体边界对（512）的序列
                attention_mask,  # 自注意力矩阵，1024*1024
                torch.tensor(position_ids),  # 位置信息，1024
                torch.tensor(labels, dtype=torch.int64),  # 标签信息，256
                torch.tensor(mention_pos),  # 实体边界对的信息，256*2
                torch.tensor(full_attention_mask), # 全注意力层？或者也可以看作是填充层，总之只有句子之间可以相互看到，或者时填充了0，1024

                #将mask的位置也传入到其中
                torch.tensor(mask_id, dtype=torch.int64),
                ]

        if self.evaluate:
            item.append(entry['example_index'])
            item.append(mentions)  # 如果是评估阶段，还要加入实体对索引和实体的边界word信息

        return item

    @staticmethod
    def collate_fn(batch):
        fields = [x for x in zip(*batch)]

        num_metadata_fields = 2
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
        # tb_writer = SummaryWriter("logs/ace_ner_logs/"+args.output_dir[args.output_dir.rfind('/'):])
        tb_writer = SummaryWriter(
            "logs/" + args.data_dir[max(args.data_dir.rfind('/'), 0):] + "_ner_logs/" + args.output_dir[
                                                                                        args.output_dir.rfind('/'):])

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    train_dataset = ACEDatasetNER(tokenizer=tokenizer,args=args)  # 其中包含每一个句子的，任意两个位置之间的实体类别信息,注:sentence中包含了几个句子，并不是一个，而每个example中的这种实体类型关系长度有限，因此可能会分割成很多块

    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(
        train_dataset)  # 对原来的数据集打乱操作
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                                  num_workers=2 * int(args.output_dir.find(
                                      'test') == -1))  # simper相当于定义了怎么从dataset里取samper，相当于一种策略，即随机取策略，
    # train_dataloader中会有batchsample参数，其实就是一个batchsize的sample所组成的集合
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(
            train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs  # //代表取整除，gradient_accumulation_steps是累计的梯度次数，每gradient_accumulation_steps次进行一次梯度更新

    # Prepare optimizer and schedule (linear warmup and decay)  warmup是预热的意思，即以一个很小的学习率逐步上升到设定的学习率，这样会使最终的收敛效果更好
    no_decay = ['bias', 'LayerNorm.weight']  # 如果model参数中出现这两个参数。则说明不需要权重衰退，bias偏置不需要正则化，LayerNorm.weight这一层也不需要正则化
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},  # weight_decay是权重衰减值，可以对参数正则化，用于防止过拟合，防止梯度爆炸
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]  # 定义优化器的参数组，包括学习率，betas，eps等

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)  # eps代表学习率的最小值
    if args.warmup_steps == -1:  # 刚开始训练时,模型的权重(weights)是随机初始化的，此时若选择一个较大的学习率,可能带来模型的不稳定(振荡)，lr先慢慢增加，超过warmup_steps时，就采用初始设置的学习率
        scheduler = get_linear_schedule_with_warmup(  # scheduler是用来调节学习率的
            optimizer, num_warmup_steps=int(0.1 * t_total), num_training_steps=t_total
            # num_warmup_steps是预热步骤数，num_training_steps是训练步数
        )
    else:
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
        )

    if args.fp16:  # 使用16bit的浮点数，可以减小缓存，但是会产生误差和溢出（pytorch默认张量精度为32位）
        try:
            from apex import amp  # 引入amp做混合精度运算
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer,
                                          opt_level=args.fp16_opt_level)  # 使用预定的opt_level对模型初始化，使用01等级
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
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)  # 优化的步骤数

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0

    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch",
                            disable=args.local_rank not in [-1, 0])  # trange(i) 是 tqdm(range(i)) 的简单写法
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)保证可以重现
    best_f1 = -1

    # 记录损失值用于后续的曲线绘制
    train_ner_loss = []
    # 记录每次训练途中的的f1指数
    train_ner_f1 = []

    for _ in train_iterator:  # 跑多少次数据集(50次)
        # if _ > 0 and (args.shuffle or args.group_edge or args.group_sort):
        #     train_dataset.initialize()
        #     if args.group_edge:
        #         train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
        #         train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=2*int(args.output_dir.find('test')==-1))
        # dataloder每次输出的就是dataset的一个batchsize大小的数据
        epoch_iterator = tqdm(train_dataloader, desc="Iteration",disable=args.local_rank not in [-1, 0])  # tqdm（）中的参数是迭代对象，而返回的对象也是一个迭代对象
        for step, batch in enumerate(epoch_iterator):  # epoch_iterator这个迭代对象的元素其实和train_dataloader的元素是一样的，step为索引，batch为一个train_dataloader中的元素，即一个batchsize的datase

            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            # batch是一个6元组,是在dataset的getitem方法中就转化成了的张量
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'position_ids': batch[2],
                      'labels': batch[3],
                      #将maskid也送入到模型中
                      'mask_id':batch[6],
                      }

            if args.model_type.find('span') != -1:
                inputs['mention_pos'] = batch[4]
            if args.use_full_layer != -1:
                inputs['full_attention_mask'] = batch[5]

            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
            # 0为损失，1为分类后的得分，后面则为bert的对应位置的输出


           #在此处进行一次计算




            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:  # 如果梯度累积步骤大于1，则需要求平均损失
                loss = loss / args.gradient_accumulation_steps

            # 在此处保存loss值，用于绘制曲线,每500次记录一个
            if (step + 1) % 147 == 0:
                train_ner_loss.append(loss.item())

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()  # 反向传播
            else:
                loss.backward()

            tr_loss += loss.item()  # 相当于取出loss的值，因为loss是个字典，里面有很多元素

            # 每隔一个梯度累积步数(gradient_accumulation_steps)做一次梯度更新
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.max_grad_norm > 0:
                    if args.fp16:  # 防止训练过程中梯度爆炸，进行梯度裁剪
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()  # 更新参数，梯度下降
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
                # 每隔logging_steps
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

                # 每隔save_steps保存一次参数，评估一次
                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    update = True
                    # Save model checkpoint
                    if args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer)
                        f1 = results['f1']

                        # 将f1指数保存到文件中，用于后续绘图,这个就是每次的评估阶段保存一次
                        train_ner_f1.append(f1)

                        tb_writer.add_scalar('f1', f1, global_step)

                        if f1 > best_f1:
                            best_f1 = f1
                            print('Best F1', best_f1)
                        else:
                            update = False

                    if update:  # Checkpoint是用于描述在每次训练后保存模型参数
                        checkpoint_prefix = 'checkpoint'
                        output_dir = os.path.join(args.output_dir, '{}-{}'.format(checkpoint_prefix, global_step))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = model.module if hasattr(model,'module') else model  # Take care of distributed/parallel training
                        model_to_save.save_pretrained(output_dir)  # 将权重保存到相关目录下

                        torch.save(args, os.path.join(output_dir, 'training_args.bin'))  # 这个只是保存命令
                        logger.info("Saving model checkpoint to %s", output_dir)

                        _rotate_checkpoints(args, checkpoint_prefix)
            # 超过最大迭代步数，则停止
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
        train_los.write(str(train_ner_loss))

    with open(f1_output_dir, 'w') as train_f:
        train_f.write(str(train_ner_f1))

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step, best_f1


def evaluate(args, model, tokenizer, prefix="", do_test=False):
    eval_output_dir = args.output_dir

    results = {}

    eval_dataset = ACEDatasetNER(tokenizer=tokenizer, args=args, evaluate=True, do_test=do_test)
    ner_golden_labels = set(eval_dataset.ner_golden_labels)
    ner_tot_recall = eval_dataset.tot_recall
    # (l_idx, n), (start, end), label) )#l_index代表是文件中的第几行，n代表这一行中的第几个句子，ner_golden_labels代表整个文件中的实体
    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    eval_sampler = SequentialSampler(eval_dataset)

    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                 collate_fn=ACEDatasetNER.collate_fn,
                                 num_workers=4 * int(args.output_dir.find('test') == -1))

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    scores = defaultdict(dict)
    predict_ners = defaultdict(list)

    model.eval()

    start_time = timeit.default_timer()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        indexs = batch[-2]  # 一个batch_size的样本的实体span的边界信息
        batch_m2s = batch[-1]  #

        batch = tuple(t.to(args.device) for t in batch[:-2])  # 除去后面的两层位置信息

        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'position_ids': batch[2],
                      #   'labels':         batch[3]
                      'mask_id':batch[6],
                      }

            if args.model_type.find('span') != -1:
                inputs['mention_pos'] = batch[4]
            if args.use_full_layer != -1:
                inputs['full_attention_mask'] = batch[5]

            outputs = model(**inputs)

            ner_logits = outputs[0]
            ner_logits = torch.nn.functional.softmax(ner_logits,dim=-1)  # 此处输出和train阶段不一样，outputs[0]就是线形层将实体映射到每个类别上的得分
            ner_values, ner_preds = torch.max(ner_logits, dim=-1)  # 得到预测的值

            for i in range(len(indexs)):  # 共有多少个实体
                index = indexs[i]  # indexs[i]的值是第几行的第几个句子，也就是第i个样本的的信息
                m2s = batch_m2s[i]  # batch_m2s[i]的值是这个样本中的所有可能的实体span的边界信息
                for j in range(len(m2s)):  # 样本中的第几个实体span
                    obj = m2s[j]
                    ner_label = eval_dataset.ner_label_list[ner_preds[i, j]]  # 看看预测出的结果是什么
                    if ner_label != 'NIL':
                        scores[(index[0], index[1])][(obj[0], obj[1])] = (
                        float(ner_values[i, j]), ner_label)  # scores[i，j)][x，y]，i，j表示句子的索引，xy表示边界信息
                        # ner_values[i,j]指的是第几个样本的第几个实体span
    cor = 0
    tot_pred = 0
    cor_tot = 0
    tot_pred_tot = 0

    for example_index, pair_dict in scores.items():
        # pair_dict指的就是在某一个确切的句子中（不是sentences，即样本），每个实体span的信息
        sentence_results = []
        for k1, (v2_score, v2_ner_label) in pair_dict.items():
            if v2_ner_label != 'NIL':
                sentence_results.append((v2_score, k1, v2_ner_label))

        sentence_results.sort(key=lambda x: -x[0])
        no_overlap = []

        def is_overlap(m1, m2):
            if m2[0] <= m1[0] and m1[0] <= m2[1]:
                return True
            if m1[0] <= m2[0] and m2[0] <= m1[1]:
                return True
            return False

        for item in sentence_results:  # sentence_results（x,y,z）x代表得分，y代表实体span的边界信息，z代表类别，注，这个只是在一个样本中的一个句子有作用
            m2 = item[1]  # 边界
            overlap = False  #
            for x in no_overlap:
                _m2 = x[1]
                if (is_overlap(m2, _m2)):  # 判断当前判断的实体是否与已经判断的有重叠
                    if args.data_dir.find('ontonotes') != -1:
                        overlap = True
                        break
                    else:

                        if item[2] == x[2]:
                            overlap = True
                            break

            if not overlap:
                no_overlap.append(item)

            pred_ner_label = item[2]
            tot_pred_tot += 1  # 累计预测的实体的个数？包括重叠和不重叠
            if (example_index, m2, pred_ner_label) in ner_golden_labels:
                cor_tot += 1  # 预测正确的个数

        for item in no_overlap:
            m2 = item[1]
            pred_ner_label = item[2]
            tot_pred += 1  # 真正的预测的实体对，去除重复的
            if args.output_results:
                predict_ners[example_index].append((m2[0], m2[1], pred_ner_label))
            if (example_index, m2, pred_ner_label) in ner_golden_labels:
                cor += 1  # 正确性

    evalTime = timeit.default_timer() - start_time
    logger.info("  Evaluation done in total %f secs (%f example per second)", evalTime, len(eval_dataset) / evalTime)

    precision_score = p = cor / tot_pred if tot_pred > 0 else 0  # 正确率
    recall_score = r = cor / ner_tot_recall  # 召回率
    f1 = 2 * (p * r) / (p + r) if cor > 0 else 0.0

    p = cor_tot / tot_pred_tot if tot_pred_tot > 0 else 0
    r = cor_tot / ner_tot_recall
    f1_tot = 2 * (p * r) / (p + r) if cor > 0 else 0.0

    results = {'f1': f1, 'f1_overlap': f1_tot, 'precision': precision_score, 'recall': recall_score}

    logger.info("Result: %s", json.dumps(results))

    if args.output_results:
        f = open(eval_dataset.file_path)
        if do_test:
            output_w = open(os.path.join(args.output_dir, 'ent_pred_test.json'), 'w')
        else:
            output_w = open(os.path.join(args.output_dir, 'ent_pred_dev.json'), 'w')
        for l_idx, line in enumerate(f):
            data = json.loads(line)
            num_sents = len(data['sentences'])
            predicted_ner = []
            for n in range(num_sents):
                item = predict_ners.get((l_idx, n), [])
                item.sort()
                predicted_ner.append(item)

            data['predicted_ner'] = predicted_ner
            output_w.write(json.dumps(data) + '\n')

    return results


import numpy as np


# 读取存储为txt文件的数据
def data_read(dir_path):
    with open(dir_path, "r") as f:
        raw_data = f.read()
        data = raw_data[1:-1].split(", ")  # [-1:1]是为了去除文件中的前后中括号"[]"

    return np.asfarray(data, float)


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default='ace_data', type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(
                            ALL_MODELS))
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
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run test on the dev set.")

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
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
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

    parser.add_argument("--train_file", default="train.json", type=str)
    parser.add_argument("--dev_file", default="dev.json", type=str)
    parser.add_argument("--test_file", default="test.json", type=str)

    parser.add_argument('--alpha', type=float, default=1, help="")
    parser.add_argument('--max_pair_length', type=int, default=256, help="")
    parser.add_argument('--max_mention_ori_length', type=int, default=8, help="")
    parser.add_argument('--lminit', action='store_true')
    parser.add_argument('--norm_emb', action='store_true')
    parser.add_argument('--output_results', action='store_true')
    parser.add_argument('--onedropout', action='store_true')
    parser.add_argument('--no_test', action='store_true')
    parser.add_argument('--use_full_layer', type=int, default=-1, help="")
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--group_edge', action='store_true')
    parser.add_argument('--group_axis', type=int, default=-1, help="")
    parser.add_argument('--group_sort', action='store_true')

    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))

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

    if args.do_train and args.local_rank in [-1, 0] and args.output_dir.find('test') == -1:
        create_exp_dir(args.output_dir,
                       scripts_to_save=['run_acener.py', 'transformers/src/transformers/modeling_bert.py',
                                        'transformers/src/transformers/modeling_albert.py'])

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)
    if args.data_dir.find('ace') != -1:
        num_labels = 8
    elif args.data_dir.find('scierc') != -1:
        num_labels = 7
    elif args.data_dir.find('ontonotes') != -1:
        num_labels = 19
    else:
        assert (False)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          num_labels=num_labels)
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)

    config.max_seq_length = args.max_seq_length
    config.alpha = args.alpha
    config.onedropout = args.onedropout
    config.use_full_layer = args.use_full_layer

    model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path),
                                        config=config)

    if args.model_type.startswith('albert'):
        special_tokens_dict = {'additional_special_tokens': ['[unused' + str(x) + ']' for x in range(4)]}
        tokenizer.add_special_tokens(special_tokens_dict)
        # print ('add tokens:', tokenizer.additional_special_tokens)
        # print ('add ids:', tokenizer.additional_special_tokens_ids)
        model.albert.resize_token_embeddings(len(tokenizer))

    if args.do_train and args.lminit:
        # 未找到roberta则跳进去
        if args.model_type.find('roberta') == -1:
            entity_id = tokenizer.encode('entity', add_special_tokens=False)
            assert (len(entity_id) == 1)
            entity_id = entity_id[0]
            mask_id = tokenizer.encode('[MASK]', add_special_tokens=False)
            assert (len(mask_id) == 1)
            mask_id = mask_id[0]
        else:
            entity_id = 10014
            mask_id = 50264

        logger.info('entity_id: %d', entity_id)
        logger.info('mask_id: %d', mask_id)

        if args.model_type.startswith('albert'):
            word_embeddings = model.albert.embeddings.word_embeddings.weight.data
            word_embeddings[30000].copy_(word_embeddings[mask_id])
            word_embeddings[30001].copy_(word_embeddings[entity_id])
        elif args.model_type.startswith('roberta'):
            word_embeddings = model.roberta.embeddings.word_embeddings.weight.data
            word_embeddings[50261].copy_(word_embeddings[mask_id])  # entity
            word_embeddings[50262].data.copy_(word_embeddings[entity_id])
        else:
            # 31090*768的张量(初始随机化)均值为0，方差为1
            # position embedding 512*768
            word_embeddings = model.bert.embeddings.word_embeddings.weight.data
            word_embeddings[1].copy_(word_embeddings[mask_id])
            word_embeddings[2].copy_(word_embeddings[entity_id])  # entity

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)
    best_f1 = 0

    # Training
    if args.do_train:
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
        # plt.ylim(0,0.5)
        plt.xlabel('iters')  # x轴标签
        plt.ylabel('loss')  # y轴标签

        # 以x_train_loss为横坐标，y_train_loss为纵坐标，曲线宽度为1，实线，增加标签，训练损失，
        # 默认颜色，如果想更改颜色，可以增加参数color='red',这是红色。
        plt.plot(x_train_loss, y_train_loss, linewidth=1, linestyle="solid", label="train loss")
        plt.legend()
        plt.title(data_dir['data_name_loss'] + 'curve')
        plt.show()

        # 绘制评估阶段的f1指数曲线
        train_f1_path = os.path.join(args.data_dir, data_dir['data_path_f1'])  # 存储文件路径# 存储文件路径

        y_train_acc = data_read(train_f1_path)  # 训练准确率值，即y轴
        x_train_acc = range(len(y_train_acc))  # 训练阶段准确率的数量，即x轴

        plt.figure()

        # 去除顶部和右边框框
        ax = plt.axes()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.xlabel('epochs')  # x轴标签
        plt.ylabel('f1')  # y轴标签

        # 以x_train_acc为横坐标，y_train_acc为纵坐标，曲线宽度为1，实线，增加标签，训练损失，
        # 增加参数color='red',这是红色。
        plt.plot(x_train_acc, y_train_acc, color='red', linewidth=1, linestyle="solid", label="train f1")
        plt.legend()
        plt.title(data_dir['data_name_f1'] + 'curve')
        plt.show()

        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)
        update = True
        if args.evaluate_during_training:
            results = evaluate(args, model, tokenizer)

            f1 = results['f1']
            if f1 > best_f1:
                best_f1 = f1
                print('Best F1', best_f1)
            else:
                update = False

        if update:
            checkpoint_prefix = 'checkpoint'
            output_dir = os.path.join(args.output_dir, '{}-{}'.format(checkpoint_prefix, global_step))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = model.module if hasattr(model,
                                                    'module') else model  # Take care of distributed/parallel training

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
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))

        logger.info("Evaluate on test set")

        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""

            # 注意：在修改完模型后，模型保存的config会是7*1536，而不是7*3072，因此可能需要修改
            model = model_class.from_pretrained(checkpoint, config=config)

            model.to(args.device)
            result = evaluate(args, model, tokenizer, prefix=global_step, do_test=not args.no_test)

            result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
            results.update(result)

    if args.local_rank in [-1, 0]:
        output_eval_file = os.path.join(args.output_dir, "results.json")
        json.dump(results, open(output_eval_file, "w"))
        logger.info("Result: %s", json.dumps(results))


if __name__ == "__main__":


    main()
