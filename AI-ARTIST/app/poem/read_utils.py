# -*- coding=utf-8 -*-
import numpy as np
import copy
import time
import tensorflow as tf
import pickle

#产生batch
def batch_generator(arr, n_seqs, n_steps):
    #arr,n_seqs:sequence个数,n_steps：一个sequence的长度
    arr = copy.copy(arr)#拷贝
    batch_size = n_seqs * n_steps#batch大小
    n_batches = int(len(arr) / batch_size)#一个epoch几个batch
    arr = arr[:batch_size * n_batches]#采样
    arr = arr.reshape((n_seqs, -1))#按照seqs进行reshape
    while True:
        np.random.shuffle(arr)#乱序
        for n in range(0, arr.shape[1], n_steps):
            x = arr[:, n:n + n_steps]
            y = np.zeros_like(x)
            y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
            #y为x右移一位
            yield x, y
            #返回x,y


class TextConverter(object):
    def __init__(self, text=None, max_vocab=5000, filename=None):
        if filename is not None:
            with open(filename, 'rb') as f:
                self.vocab = pickle.load(f)#取回vocab
        else:
            vocab = set(text)
            print(len(vocab))
            # max_vocab_process
            vocab_count = {}
            for word in vocab:
                vocab_count[word] = 0
            for word in text:
                vocab_count[word] += 1
            #单词数量
            vocab_count_list = []
            for word in vocab_count:
                vocab_count_list.append((word, vocab_count[word]))
            vocab_count_list.sort(key=lambda x: x[1], reverse=True)
            #排序
            if len(vocab_count_list) > max_vocab:
                vocab_count_list = vocab_count_list[:max_vocab]
            vocab = [x[0] for x in vocab_count_list]
            self.vocab = vocab
            #词典（5000字）
        #词to下标
        self.word_to_int_table = {c: i for i, c in enumerate(self.vocab)}
        #enumerate:[0]为下标,[1]为值
        self.int_to_word_table = dict(enumerate(self.vocab))
        #下标to词
        
        
    """工具"""   
    @property
    
    #功能如其名
    #词典大小
    def vocab_size(self):
        return len(self.vocab) + 1

    def word_to_int(self, word):
        if word in self.word_to_int_table:
            return self.word_to_int_table[word]
        else:
            return len(self.vocab)

    #对超出词汇做unk处理
    def int_to_word(self, index):
        if index == len(self.vocab):
            return '<unk>'
        elif index < len(self.vocab):
            return self.int_to_word_table[index]
        else:
            raise Exception('Unknown index!')

    #text为原诗汉字表示，现通过np转化为数组表示
    def text_to_arr(self, text):
        arr = []
        for word in text:
            arr.append(self.word_to_int(word))
        return np.array(arr)
    #将数组诗词转为汉字诗词
    def arr_to_text(self, arr):
        words = []
        for index in arr:
            words.append(self.int_to_word(index))
        return "".join(words)
    #词典打包保存本地
    def save_to_file(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.vocab, f)
