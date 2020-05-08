#!/usr/bin/env python
# coding: utf-8

#!-*- encoding=utf-8 -*-
from keras.models import Sequential, Model
from keras.layers import *
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import Callback

import numpy as np
import random
import sys
import os
import json
import tensorflow as tf
import keras.backend as K

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = tf.ConfigProto()
config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
sess = tf.Session(config=config)

# 占位符
pad_token = 0
oov_token = 1
start_token = 2
end_token = 3

# 数据处理设置
max_len = 50
min_count = 1
voc_size = 100000
batch_size = 32

# 模型参数
dp_dis = 0.5 #判别器的dropout大小
emb_size = 64
gru_dim = 512 #enc、dec的GRU大小
z_dim = 64 #encode向量大小
c_dim = 2 #大小为2，因为有两类
alpha = 1e-1 #判别器的loss的weight
lr = 1e-3
rational_alpha = 0.9
rational_weight = 0.01

# 训练集以及vocab
ham_path = '../data/dataset/sms/ham_train.txt'
spam_path = '../data/dataset/sms/spam_train.txt'
vocab_name = 'sms-vocab.json'

# 参数存储地址
# alpha大小-loss权重
enc_path = '../model/enc-sms-attn-vae-mask-'+str(rational_alpha)+'-'+str(rational_weight)+'-50000.h5'
dec_path = '../model/dec-sms-attn-vae-mask-'+str(rational_alpha)+'-'+str(rational_weight)+'-50000.h5'
dis_path = '../model/dis-sms-attn-vae-mask-'+str(rational_alpha)+'-'+str(rational_weight)+'-50000.h5'

if os.path.exists(vocab_name):
    id2char,char2id = json.load(open(vocab_name))
    id2char = {int(i):j for i,j in id2char.items()}
else:
    char_freq = {}

    for d in train:
        c = d[0]
        for char in c:
            if char not in char_freq.keys():
                char_freq[char] = 1
            else:
                char_freq[char] += 1

    print('whole number of distinct chars is', len(char_freq))

    char_freq = {i:j for i,j in char_freq.items() if j >= min_count}
    print('whole number of distinct chars more frequent than %d is %d'%(min_count, len(char_freq)))
    
    sort_char_freq = sorted(char_freq.items(), key=lambda x:x[1], reverse=True)
    sort_char_freq = sort_char_freq[:voc_size]

    char2id = {}
    #0:pad
    #1:unk
    char2id['<pad>'] = 0
    char2id['<unk>'] = 1
    char2id['<bos>'] = 2
    char2id['<eos>'] = 3
    
    for a in sort_char_freq:
        char2id[a[0]] = len(char2id) 
    id2char = {j:i for i,j in char2id.items()}
    
    json.dump([id2char, char2id], open(vocab_name, 'w'))
    
print('load vocab successfully')



# coding: utf-8
import numpy as np
import os
import pickle
import codecs
import sys
sys.setrecursionlimit(10000)


class TrieNode:
    def __init__(self):
        self.success = dict()  # 转移表
        self.failure = None  # 错误表
        self.emits = set()  # 输出表


class CreateAcAutomaton(object):

    def __init__(self, patterns, save_path="  "):
        """
        :param patterns:  模式串列表
        :param save_path:   AC自动机持久化位置
        """
        self._savePath = save_path.strip()
        assert isinstance(self._savePath, str) and self._savePath != ""
        self._patterns = patterns
        self._root = TrieNode()
        self.__insert_node()
        self.__create_fail_path()
        
    def __insert_node(self):
        """
        Create Trie
        """
        for pattern in self._patterns:
            line = self._root
            for character in pattern:
                line = line.success.setdefault(character, TrieNode())
            line.emits.add(pattern)

    def __create_fail_path(self):
        """
        Create Fail Path
        """
        my_queue = list()
        for node in self._root.success.values():
            node.failure = self._root
            my_queue.append(node)
        while len(my_queue) > 0:
            gone_node = my_queue.pop(0)
            for k, v in gone_node.success.items():
                my_queue.append(v)
                parent_failure = gone_node.failure

                while parent_failure and k not in parent_failure.success.keys():
                    parent_failure = parent_failure.failure
                v.failure = parent_failure.success[k] if parent_failure else self._root
                if v.failure.emits:
                    v.emits = v.emits.union(v.failure.emits)

    def __save_corasick(self):
        with codecs.open(self._savePath, "wb") as f:
            pickle.dump(self._root, f)

    def __load_corasick(self):
        with codecs.open(self._savePath, "rb") as f:
            return pickle.load(f)

    def search(self, context):
        """"""
        search_result = list()
        search_node = self._root

        index = -1
        for char in context:
            index += 1
            while search_node and char not in search_node.success.keys():
                search_node = search_node.failure
            if not search_node:
                search_node = self._root
                continue
            search_node = search_node.success[char]
            if search_node.emits:
                for s in search_node.emits:
                    search_result.append([index-len(s)+1, index, s])
        return search_result

rationales = open("../data/rationales", 'r').readlines()
rationales = [x.strip() for x in rationales]
ct = CreateAcAutomaton(rationales, "model.pkl")

def gen_mask(text, ct, max_len=50, alpha=0.8):

    if len(text) > max_len:
        text = text[:max_len]

    weights = np.ones(shape=[1, len(text), 1])
    st = ct.search(text)

    for item in st:
        for i in range(item[0], item[1]+1):
            weights[0][i][0] = max(0, weights[0][i][0]-alpha)
    return np.pad(weights/np.sum(weights), ((0,0),(0, max_len-len(text)),(0,0)), 'constant', constant_values=0)

def str2id(s, start_end = False):
    # split后的list转整数id
   # 补上<BOS>1和<EOS>2标记,找不到就用<UNK>3
    ids = [char2id.get(c, oov_token) for c in s]
    if start_end:
        ids = [start_token] + ids + [end_token]
    return ids

def padding(x,y,z):
    # padding至batch内的最大长度
    # vae有三个输入，encode
    # x和y因为要一样长
    # 卧槽，这样就能每一个batch不用一样的max_len
   # ml = max([len(i) for i in x+y+z])
    ml = max_len
    x = [i + [0] * (ml-len(i)) for i in x]
    y = [i + [0] * (ml+2-len(i)) for i in y]
    z = [i + [0] * (ml+2-len(i)) for i in z]
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    return x,y,z
    
def train_generator(data):
    x = []
    y = []
    z = []
    Mask = []
    labels = []
    
    while True:
        np.random.shuffle(data)    
        for d in data:
            #对每一个data单独处理
            text, label = d[0], d[1]
            text = text[:max_len]
            mask = gen_mask(text, ct, alpha=rational_alpha)[0,:,:]
            Mask.append(mask)
            text_enc = str2id(text, start_end=False)
            text_dec = str2id(text, start_end=True)
            x.append(text_enc)
            y.append(text_dec)
            z.append(text_dec[1:])
            l = [0, 0]
            l[int(label)] = 1
            labels.append(l)
            if len(x) == batch_size:
                x,y,z = padding(x, y, z)
                labels=np.array(labels)
                Mask = np.array(Mask)
                yield [x,y,z,labels, Mask], None
                x = []
                y = []   
                z = []
                Mask = []
                labels = []
                
                
def sample(preds, diversity=1.0):
    # sample from te given prediction
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / diversity
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)

    return np.argmax(probas)

def sample_batch(preds_batch, diversity=1.0):
    # 支持batch式sample
    result = []
    for pred in preds_batch:
        result.append(sample(pred, diversity=diversity))
    
    return result

def argmax(preds):
    preds = np.asarray(preds).astype('float64')
    return np.argmax(preds)

def argmax_batch(preds_batch):
    # 支持batch式argmax
    result = []
    for preds in preds_batch:
        result.append(argmax(preds))
    
    return result

#Model 部分
#encoder部分
#接受一个maxlen的句子，返回一个fix-length vector
encoder_input = Input(shape=(max_len, ), dtype='int32')
c_in = Input(shape=(c_dim, ))
emb_layer = Embedding(len(char2id), emb_size)
encoder_emb = emb_layer(encoder_input) # id转向量
encoder = Bidirectional(GRU(gru_dim//2,return_sequences=True))
encoder_h_seq = encoder(encoder_emb)

# Attention-Pooling
attn_h_seq = Dense(1)(encoder_h_seq) #[bs,ml,1]
enc_input_mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(encoder_input) #[bs,ml,1]
enc_input_mask = Lambda(lambda x: (-1e10)*(1-x))(enc_input_mask)
attn_h_seq = Lambda(lambda x:x[0]+x[1])([enc_input_mask, attn_h_seq])
attn_h_seq = Lambda(lambda x: K.expand_dims(K.softmax(K.squeeze(x, axis=2), axis=1), axis=2))(attn_h_seq) 
encoder_h = Lambda(lambda x:K.sum(x[0]*x[1], axis=1))([attn_h_seq, encoder_h_seq])

# 强制学习到的Attn分布和Rational分布接近
mask_input = Input(shape=(max_len, 1))
rational_loss = Lambda(lambda x: K.sum(x[0]*K.log((x[0]+1e-10)/(x[1]+1e-10))))([attn_h_seq, mask_input])

# 输出attn可视化
get_enc_attn = K.function([encoder_input], [attn_h_seq])

# 算均值方差
mean_layer = Dense(z_dim)
var_layer = Dense(z_dim)

encoder_ch = Concatenate(axis=-1)([encoder_h, c_in])
z_mean = mean_layer(encoder_ch)
z_log_var = var_layer(encoder_h)

kl_loss = Lambda(lambda x: K.mean(- 0.5 * K.sum(1 + x[0] - K.square(x[1]) - K.exp(x[0]), axis=-1)))([z_log_var, z_mean])

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], z_dim), mean=0, stddev=1)
    return z_mean + K.exp(z_log_var / 2) * epsilon


enc_z = Lambda(sampling)([z_mean, z_log_var])
enc_z = Lambda(lambda x: K.in_train_phase(x[0], x[1]))([enc_z, z_mean])

enc_model = Model([encoder_input, c_in, mask_input], [enc_z, kl_loss, rational_loss])
enc_model.load_weights(enc_path)
print('enc权重加载成功')
#enc_model.summary()



# decoder
decoder_z_input = Input(shape=(z_dim, ))
decoder_c_input = Input(shape=(c_dim, ))
decoder_input = Input(shape=(max_len+2, ))

decoder_dense = Dense(emb_size)
decoder_emb = emb_layer(decoder_input)
dec_softmax = Dense(len(char2id), activation='softmax')


decoder_z = RepeatVector(max_len+2)(decoder_z_input)
decoder_c = RepeatVector(max_len+2)(decoder_c_input)
decoder_z = Concatenate(axis=-1)([decoder_z, decoder_c])

decoder_h = Concatenate(axis=-1)([decoder_emb, decoder_z])

decoder_h = GRU(gru_dim, return_sequences=True)(decoder_h)

decoder_h = decoder_dense(decoder_h)
decoder_output = dec_softmax(decoder_h)

dec_model = Model([decoder_input, decoder_z_input, decoder_c_input], decoder_output)
dec_model.load_weights(dec_path)
print('dec权重加载成功')
#dec_model.summary()



x_in = Input(shape=(max_len, ))
x_emb = emb_layer(x_in)
conv1 = Conv1D(100, 3, activation='relu')
conv2 = Conv1D(100, 4, activation='relu')
conv3 = Conv1D(100, 5, activation='relu')
x_conv1 = conv1(x_emb)
x_pool1 = GlobalMaxPooling1D()(x_conv1)
x_conv2 = conv2(x_emb)
x_pool2 = GlobalMaxPooling1D()(x_conv2)
x_conv3 = conv3(x_emb)
x_pool3 = GlobalMaxPooling1D()(x_conv3)
x_h = Concatenate(axis=-1)([x_pool1, x_pool2, x_pool3])
x_h = Dropout(dp_dis)(x_h)
x_output = Dense(c_dim, activation='softmax')(x_h)
dis_model = Model(x_in, x_output)
dis_model.load_weights(dis_path)
print('dis权重加载成功')
#dis_model.summary()


def id2str(ids):
    return [id2char[x] for x in ids]


def gen_from_vec(diversity, z, c, argmax_flag):
    start_index = start_token #<BOS>
    start_word = id2char[start_index]
    gen_sen = []
    generated = [[start_index]]
    
    while(end_token not in generated[0] and len(generated[0]) <= (max_len+2)):
        x_seq = pad_sequences(generated, maxlen=max_len+2,padding='post')
        preds = dec_model.predict([x_seq, z, c], verbose=0)[0]
        preds = preds[len(generated[0])-1][3:]
        if argmax_flag:
            next_index = argmax(preds)
        else:
            next_index = sample(preds, diversity)
        next_index += 3
        next_word = id2char[next_index]
        gen_sen.append(next_word)

        generated[0] += [next_index]
        
    if gen_sen[-1] == '<eos>':
        gen_sen = gen_sen[:-1]
        
    return gen_sen

def gen_from_vec_batch(batch_size, diversity, z, c, argmax_flag):
    # 支持batch式生成，以加快生成速度
    # 要求z和c大小都是[batch_size, dim]
    gen_sen = []
    generated = []
    for i in range(batch_size):
        generated.append([start_token])
        gen_sen.append([])
    
    for i in range(max_len+1):
        x_seq = pad_sequences(generated, maxlen=max_len+2,padding='post')
        preds = dec_model.predict([x_seq, z, c], verbose=0) #[bs, ml, V]
        preds = preds[:, len(generated[0])-1, 3:]
        if argmax_flag:
            next_index_batch = argmax_batch(preds)
        else:
            next_index_batch = sample_batch(preds, diversity)
            
        next_index_batch = [next_index+3 for next_index in next_index_batch]
        next_word_batch = [id2char[next_index] for next_index in next_index_batch]
        
        for index, next_word in enumerate(next_word_batch):
            gen_sen[index].append(next_word)
            generated[index] += [next_index_batch[index]]
            
    # 去掉eos符号
    new_gen_sen = []
    for gen_s in gen_sen:
        if '<eos>' in gen_s:
            end_index = gen_s.index('<eos>')
            new_gen_sen.append(gen_s[:end_index-1])
        else:
            new_gen_sen.append(gen_s)
        
    return new_gen_sen


ham_path = '../data/dataset/sms/ham_train.txt'
spam_path = '../data/dataset/sms/spam_train.txt'

#读取train的数据
train = []

with open(ham_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip().lower()
        text = line.split(' ')
        train.append([text, 1])
        
with open(spam_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip().lower()
        text = line.split(' ')
        train.append([text, 0])
        
np.random.shuffle(train)

test_gen = train_generator(train)

def DA_AttMaskVAE(text, num=1, temp=1):
    '''
    输入
    text：文本，字list表示（如 ['你', '好']（只支持传一个文本）
    num：生成的样本数
    temp：sample时添加的控制因子，一般而言 temp越小 多样性越差，temp越大，多样性越好
    
    返回
    大小为num的text list，同样用字的list代表文本
    '''
    text = text[:max_len]
    
    # 通过rational得到的mask分布
    mask = gen_mask(text, ct, alpha=rational_alpha)
    mask = np.array(mask)
    
    text = str2id(text, start_end=False)
    text = np.array([text])
    
    text = pad_sequences(text, maxlen=max_len, padding='post', truncating='post') 
    
    un_c = dis_model.predict(text)
    un_c = un_c
    un_c[un_c>0.5] = 1
    un_c[un_c<0.5] = 0
    un_c = un_c.astype(np.int32)

    un_enc_z,_,_ = enc_model.predict([text, un_c, mask])

    # DA部分
    da_sen = []
    for i in range(num):   
        da_sen.append(gen_from_vec(temp, un_enc_z, un_c, False))
    
    return da_sen

def Get_Enc_Attn(text):
    # 输入：文本
    # 返回Attention List
    
    text = text[:max_len]
    text = str2id(text, start_end=False)
    text = np.array([text])
    text = pad_sequences(text, maxlen=max_len, padding='post', truncating='post')    
    attn = get_enc_attn([text])
    
    return attn
       
def DA_AttMaskVAE_batch(text_batch, num=1, temp=1):
    '''
    输入
    text_batch：batch文本的list of list表示，第二层list为字的列表 如[['你', '好'], ['我', '好']]
    num：每个文本生成的样本数
    temp：sample时添加的控制因子，一般而言 temp越小 多样性越差，temp越大，多样性越好
    
    返回
    多维list，从前到后的大小分别为 num，text_batch中的text数，生成的字list
    如 num=3时 返回 [[['你', '好'], ['我', '好']], [['你', '好'], ['我', '好']], [['你', '好'], ['我', '好']]]
    '''
    text_input_batch = []
    mask_batch = []
    
    for text in text_batch:
        text = text[:max_len]

        # 通过rational得到的mask分布
        mask = gen_mask(text, ct, alpha=rational_alpha)
        mask = np.array(mask)
        mask_batch.append(mask)

        text = str2id(text, start_end=False)
        text = np.array([text])

        text = pad_sequences(text, maxlen=max_len, padding='post', truncating='post') 
        
        text_input_batch.append(text)
        
    text_input_batch = np.squeeze(np.array(text_input_batch), axis=1)
    mask_batch = np.squeeze(np.array(mask_batch), axis=1)

    un_c = dis_model.predict(text_input_batch)
    un_c[un_c>0.5] = 1
    un_c[un_c<0.5] = 0
    un_c = un_c.astype(np.int32)

    un_enc_z,_,_ = enc_model.predict([text_input_batch, un_c, mask_batch])

    # DA部分
    da_sen = []
    for i in range(num):   
        da_sen.append(gen_from_vec_batch(len(text_batch), temp, un_enc_z, un_c, False))
    
    return da_sen

if __name__ == "__main__":

    # 占位符
    pad_token = 0
    oov_token = 1
    start_token = 2
    end_token = 3

    # 数据处理设置
    max_len = 100
    min_count = 1
    voc_size = 100000

    batch_size = 32

    # 模型参数
    dp_dis = 0.5 #判别器的dropout大小
    emb_size = 64
    gru_dim = 512 #enc、dec的GRU大小
    z_dim = 64 #encode向量大小
    c_dim = 2 #大小为2，因为有两类
    alpha = 1e-1 #判别器的loss的weight
    lr = 1e-3

    # 训练集以及vocab
    ham_train_path = './data/dataset/sms/ham_train.txt'
    spam_train_path = './data/dataset/sms/spam_train.txt'
    # spam_train_path = './data/dataset/sms/spam_train_5.txt'
    # spam_train_path = './data/dataset/sms/spam_train_10.txt'
    # spam_train_path = './data/dataset/sms/spam_train_100.txt'

    ham_test_path = './data/dataset/sms/ham_test.txt'
    spam_test_path = './data/dataset/sms/spam_test.txt'

    vocab_name = './data/dataset/sms/vocab.json'

    num = 10
    spam_da_path = './data/dataset/sms/spam_train_amvae_da.txt'
    with open(spam_da_path, 'w', encoding='utf-8') as f_out:
        with open(spam_train_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            lines = [line.split(' ') for line in lines]
            data_aug = DA_AttMaskVAE_batch(lines, num)
            for j in range(len(lines)):
                for i in range(num):
                    f_out.write(' '.join(data_aug[i][j]) + '\n')
                f_out.write('===========================================================' + '\n')

    num = 10
    ham_da_path = './data/dataset/sms/ham_train_amvae_da.txt'
    with open(ham_da_path, 'w', encoding='utf-8') as f_out:
        with open(ham_train_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            lines = [line.split(' ') for line in lines]
            data_aug = DA_AttMaskVAE_batch(lines, num)
            for j in range(len(lines)):
                for i in range(num):
                    f_out.write(' '.join(data_aug[i][j]) + '\n')
                f_out.write('===========================================================' + '\n')