import os
import random
import json
import torch
import argparse
import numpy as np
import networkx as nx


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def get_batch(batch, word_vec, emb_dim = 300):
    # sent in batch in decreasing order of lengths (bsize, max_len, word_dim)
    lengths = np.array([len(x) for x in batch])
    max_len = np.max(lengths)
    embed = np.zeros((max_len, len(batch), emb_dim))

    for i in range(len(batch)):
        for j in range(len(batch[i])):
            embed[j, i, :] = word_vec[batch[i][j]]
    return torch.from_numpy(embed).float(), lengths


def get_word_dict(sentences):
    '''create vocab of words'''
    word_dict = {}
    for sent in sentences:
        for word in sent.split():
            if word not in word_dict:
                word_dict[word] = ''
    word_dict['<s>'] = ''
    word_dict['</s>'] = ''
    word_dict['<p>'] = ''
    return word_dict


def get_glove(word_dict, glove_path):
    '''create word_vec with glove vectors'''
    word_vec = {}
    with open(glove_path, encoding='utf8') as f:
        for line in f:
            word, vec = line.split(' ', 1)
            if word in word_dict:
                word_vec[word] = np.array(list(map(float, vec.split())))
    print('Found {0}(/{1}) words with glove vectors'.format(
        len(word_vec), len(word_dict)))
    return word_vec


def build_vocab(sentences, glove_path):
    word_dict = get_word_dict(sentences)
    word_vec = get_glove(word_dict, glove_path)
    print('Vocab size : {0}'.format(len(word_vec)))
    return word_vec


def turn_graphid_2_content(total_id_list, isCV=True):
    ret_list = []
    word_tokenizer = cut

    if isCV:
        f_path = "data/step1_data/exp_morethan_50_graph/user.json"
    else:
        f_path = "data/step1_data/exp_morethan_50_graph/jd.json"
    f = open(f_path, 'r', encoding='utf8')
    f_dict = json.load(f)

    for ids_line in total_id_list:
        ids_list = []
        single_id_list = ids_line.split(' ')
        for id in single_id_list:
            if(id in f_dict):
                content_list = f_dict[id]
            else:
                continue
            content = ""
            for z in content_list:
                content += z
            content_list = list(word_tokenizer(content))
            content = ""
            for z in content_list:
                content = content + z + ' '
            ids_list.append(content)
        ret_list.append(ids_list)
    return ret_list

def data_masks(all_usr_pois, item_tail):
    us_lens = [len(upois) for upois in all_usr_pois]
    len_max = max(us_lens)
    us_pois = [upois.tolist() + item_tail * (len_max - le) for upois, le in zip(all_usr_pois, us_lens)]
    us_masks = [[1] * le + [0] * (len_max - le) for le in us_lens]

    return us_pois, us_masks, len_max
