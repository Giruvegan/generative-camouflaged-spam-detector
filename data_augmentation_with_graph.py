# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import math
import random
from copy import deepcopy
import numpy as np

np.random.seed(42)


def init_cchin():
    cchin = pickle.load(open('../data/graph/cchin/CCHIN_NX.pkl', 'r'))
    print("cchin node numbers: ")
    print(cchin.number_of_nodes())
    print("cchin edge numbers: ")
    print(cchin.number_of_edges())
    return cchin


# 通过随机游走进行数据增强
def gen_sequence_random_walking(cchin, sequence, keep_ratio=0.5, max_degree=3):
    '''

    :param sequence: input sequence, like ['一', '天', '到', '晚']
    :param keep_ratio: ratio that stay in original sequence
    :param max_degree: max walking step per node
    :return: a sequence after random walking
    '''
    new_sequence = deepcopy(sequence)
    change_num = int(math.floor(len(sequence) * (1 - keep_ratio)))
    change_indexs = random.sample(range(len(sequence)), change_num)
    # print 'change index at '
    # print change_indexs
    for index in change_indexs:
        degree = np.random.randint(low=1, high=max_degree + 1)
        # print 'choose max walking step ' + str(degree) + ' at index ' + str(index)
        # print 'original character is ' + sequence[index]
        node_from = sequence[index]
        for step in range(1, degree + 1):
            node_to = random_walking_per_node(cchin, node_from)
            print('step ' + str(step) + ' : from ' + node_from + ' to ' + node_to)
            node_from = node_to
        new_sequence[index] = node_from
    print('original sequence is ' + ' '.join(sequence))
    print('new sequence is ' + ' '.join(new_sequence))
    return new_sequence


def random_walking_per_node(cchin, node, alpha=1.0):
    if node not in cchin.adj:
        return node
    char_prob = {node: alpha}
    for char in list(cchin.adj[node]):
        edge_dict = cchin.adj[node][char]
        dict_size = len(edge_dict)
        weight = 0
        for i in range(dict_size):
            edge_type = cchin.adj[node][char][i]['type']
            edge_weight = cchin.adj[node][char][i]['weight']
            if edge_type == 'pinyin':
                if edge_weight > 0.5:
                    weight += edge_weight
            if edge_type == 'zheng':
                if edge_weight > 0.6:
                    weight += edge_weight
            if edge_type == 'stroke':
                if edge_weight > 0.8:
                    weight += edge_weight
        if weight > 0:
            char_prob[char] = weight
    weight_sum = sum(char_prob.values())
    for key in char_prob:
        char_prob[key] = (char_prob[key] / weight_sum)
    return random_pick(char_prob.keys(), char_prob.values())


def random_pick(some_list, probabilities):
    x = random.uniform(0, 1)
    cumulative_probability = 0.0
    for item, item_probability in zip(some_list, probabilities):
        cumulative_probability += item_probability
        if x < cumulative_probability:
            break
    return item


def weight_to_distance(weight):
    return 1 / (float(weight) + 0.01)


# 通过搜索进行数据增强
def gen_sequences_by_search(cchin, sequence, cost=1, max_batch_size=10, max_degree=3):
    '''

    :param sequence: input sequence, like ['一', '天', '到', '晚']
    :param cost: search cost constrain: edge cost = 1 - similarity + 0.01
    :param max_batch_size: max sequence number returned
    :param max_degree: max walking step per node
    :return:
    '''
    new_sequence = deepcopy(sequence)
    index_cost = []
    for index in range(len(sequence)):
        index_cost.append({})
        index_cost, _ = bfs_search_per_node(index, sequence[index], index_cost, cchin, max_degree, [])

    # print 'original sequence is ' + ' '.join(sequence)
    # print 'new sequence is ' + ' '.join(new_sequence)
    # return new_sequence
    print(index_cost)


def bfs_search_per_node(index, node, index_cost, cchin, max_degree, visited):
    if max_degree == 0:
        return index_cost, visited
    for char in list(cchin.adj[node]):
        if char in visited:
            continue
        edge_dict = cchin.adj[node][char]
        dict_size = len(edge_dict)
        weight = 0
        for i in range(dict_size):
            edge_type = cchin.adj[node][char][i]['type']
            edge_weight = cchin.adj[node][char][i]['weight']
            if edge_type == 'pinyin':
                if edge_weight > 0.5:
                    weight += edge_weight
            if edge_type == 'zheng':
                if edge_weight > 0.6:
                    weight += edge_weight
            if edge_type == 'stroke':
                if edge_weight > 0.8:
                    weight += edge_weight
        if weight > 0:
            index_cost[index][char] = weight_to_distance(weight)
            visited.append(char)

    for char in visited:
        index_cost, visited = bfs_search_per_node(index, char, index_cost, cchin, max_degree - 1, visited)

    return index_cost, visited


def load_frequency_prior():
    frequency_prior = {}
    with open("./data/graph/cchin/alitx_char_count.txt") as f:
        for line in f:
            (key, val) = line.split()
            frequency_prior[key] = int(val)
    return frequency_prior


# 通过字频信息来随机游走，进行数据增强
def gen_sequence_random_walking_with_frequency_prior(cchin, sequence, frequency_prior, keep_ratio=0.5, max_degree=3,
                                                     alpha=1.0):
    '''

    :param sequence: input sequence, like ['一', '天', '到', '晚']
    :param keep_ratio: ratio that stay in original sequence
    :param max_degree: max walking step per node
    :return: a sequence after random walking
    '''
    new_sequence = deepcopy(sequence)
    change_num = int(math.floor(len(sequence) * (1 - keep_ratio)))
    change_indexs = random.sample(range(len(sequence)), change_num)
    # print 'change index at '
    # print change_indexs
    for index in change_indexs:
        degree = np.random.randint(low=1, high=max_degree + 1)
        # print 'choose max walking step ' + str(degree) + ' at index ' + str(index)
        # print 'original character is ' + sequence[index]
        node_from = sequence[index]
        for step in range(1, degree + 1):
            node_to = random_walking_per_node_with_frequency_prior(cchin, node_from, frequency_prior, alpha=alpha)
            # print 'step ' + str(step) + ' : from ' + node_from + ' to ' + node_to
            node_from = node_to
        new_sequence[index] = node_from
    print('original sequence is ' + ' '.join(sequence))
    print('new sequence is ' + ' '.join(new_sequence))
    return new_sequence


def random_walking_per_node_with_frequency_prior(cchin, node, frequency_prior, alpha=1.0):
    if node not in cchin.adj:
        return node
    char_prob = {node: alpha}
    for char in list(cchin.adj[node]):
        edge_dict = cchin.adj[node][char]
        dict_size = len(edge_dict)
        weight = 0
        for i in range(dict_size):
            edge_type = cchin.adj[node][char][i]['type']
            edge_weight = cchin.adj[node][char][i]['weight']
            if edge_type == 'pinyin':
                if edge_weight > 0.5:
                    weight += edge_weight
            if edge_type == 'zheng':
                if edge_weight > 0.6:
                    weight += edge_weight
            if edge_type == 'stroke':
                if edge_weight > 0.8:
                    weight += edge_weight
        if weight > 0:
            char_prob[char] = weight
    for key in char_prob:
        if key in frequency_prior:
            char_prob[key] = char_prob[key] * frequency_prior[key]
    weight_sum = sum(char_prob.values())
    for key in char_prob:
        char_prob[key] = (char_prob[key] / weight_sum)
    return random_pick(char_prob.keys(), char_prob.values())


frequent_chars = set()
for line in open("../data/graph/cchin/alitx_char_count.txt", 'r').readlines()[:4000]:
    items = line.split(' ')
    frequent_chars.add(items[0])

close_chars = {}
for line in open("../data/graph/cchin/cchin_edges", 'r').readlines():
    items = line.strip().split(' ')
    if len(items) == 4:
        if items[0] in frequent_chars and items[1] in frequent_chars:
            if items[0] in close_chars:
                close_chars[items[0]].add(items[1])
            else:
                close_chars[items[0]] = set()

            if items[1] in close_chars:
                close_chars[items[1]].add(items[0])
            else:
                close_chars[items[1]] = set()


def GDA(sequence, num=10, keep_ratio=0.4):
    result = []
    for i in range(num):

        new_sequence = deepcopy(sequence)
        change_num = int(math.floor(len(sequence) * (1 - keep_ratio)))
        change_indexs = random.sample(range(len(sequence)), change_num)

        for index in change_indexs:
            c = new_sequence[index]
            if c not in close_chars:
                continue
            else:
                close_set = close_chars[new_sequence[index]]
                if close_set is None or len(close_set) == 0:
                    continue
                new_char = random.sample(close_set, 1)
                new_sequence[index] = new_char[0]

        result.append(new_sequence)

    return result


if __name__ == '__main__':

    # 基于图生成
    # spam_gda_path = '../data/dataset/sms/spam_train_g_da_merge.txt'
    # spam_train_path = '../data/dataset/sms/spam_train.txt'
    # with open(spam_gda_path, 'w', encoding='utf-8') as f_out:
    #     with open(spam_train_path, 'r', encoding='utf-8') as f:
    #         lines = f.readlines()
    #         print(len(lines))
    #         idx = 0
    #         for line in lines:
    #             idx += 1
    #             if idx%1000==0:
    #                 print(idx)
    #             f_out.write(line)
    #             data_aug = GDA(line.strip().split(' '), 10, 0.8)
    #             for d in data_aug:
    #                 f_out.write(' '.join(d) + '\n')
    #             f_out.write('===========================================================' + '\n')
    #
    # ham_gda_path = '../data/dataset/sms/ham_train_g_da_merge.txt'
    # ham_train_path = '../data/dataset/sms/ham_train.txt'
    # with open(ham_gda_path, 'w', encoding='utf-8') as f_out:
    #     with open(ham_train_path, 'r', encoding='utf-8') as f:
    #         lines = f.readlines()
    #         print(len(lines))
    #         idx = 0
    #         for line in lines:
    #             idx += 1
    #             if idx % 1000 == 0:
    #                 print(idx)
    #             f_out.write(line)
    #             data_aug = GDA(line.strip().split(' '), 10, 0.8)
    #             for d in data_aug:
    #                 f_out.write(' '.join(d) + '\n')
    #             f_out.write('===========================================================' + '\n')


    # 给安全部生成数据
    # spam_gda_path = '../data/dataset/化学品词短扩展结果.txt'
    # spam_train_path = '../data/dataset/化学品词短.txt'
    # with open(spam_gda_path, 'w', encoding='utf-8') as f_out:
    #     with open(spam_train_path, 'r', encoding='utf-8') as f:
    #         lines = f.readlines()
    #         print(len(lines))
    #         idx = 0
    #         for line in lines:
    #             idx += 1
    #             if idx%1000==0:
    #                 print(idx)
    #             f_out.write(line)
    #             data_aug = GDA(list(line.strip()), 10, 0.4)
    #             for d in data_aug:
    #                 f_out.write(''.join(d) + '\n')
    #             f_out.write('===========================================================' + '\n')

    # AMVAE + 图后处理生成
    f = open('../data/dataset/sms/ham_train.txt', 'r').readlines()
    f1 = open('../data/dataset/sms/ham_train_amvae_da.txt', 'r').readlines()
    f11 = open('../data/dataset/sms/ham_train_amvae_g_da_merge.txt', 'w')

    print(len(f))
    for i in range(len(f)):
        if i % 1000 == 0:
            print(i)
        f11.write(f[i])
        for j in range(11 * i, 11 * (i + 1) - 1):
            data_aug = GDA(f1[j].strip().split(' '), 1, 0.8)
            f11.write(' '.join(data_aug[0]) + '\n')
        f11.write(f1[11 * (i + 1) - 1])
    f11.close()