# -*- coding=utf-8 -*-
from __future__ import print_function
import tensorflow as tf
import numpy as np
import time
import os
from read_utils import TextConverter


"""
测试
"""

converter = TextConverter(filename='app/poem/model/poetry/converter.pkl')



#预测
def pick_top_n(preds, vocab_size, top_n=5):
    #np.squeeze:压缩preds维度，与原数组数值上无差别
    p = np.squeeze(preds)
    
    #argsort:对p升序排序，保留下标,np.argsort(p)[:-top_n],选中除了top_n大小以外的数
    # 将除了top_n个预测值的位置都置为0
    p[np.argsort(p)[:-top_n]] = 0
    # 归一化概率
    p = p / np.sum(p)
    # 随机选取一个字符
    # 在归一化的p中，随机选取一个数
    c = np.random.choice(vocab_size, 1, p=p)[0]
    return c

#模型搭建与初始化
class CharRNN:
    def __init__(self, num_classes, num_seqs=64, num_steps=50,
                 lstm_size=128, num_layers=2, learning_rate=0.001,
                 grad_clip=5, sampling=False, train_keep_prob=0.5, use_embedding=False, embedding_size=128):
        if sampling is True:
            num_seqs, num_steps = 1, 1
        else:
            num_seqs, num_steps = num_seqs, num_steps

        self.num_classes = num_classes
        self.num_seqs = num_seqs
        self.num_steps = num_steps
        self.lstm_size = lstm_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.grad_clip = grad_clip
        self.train_keep_prob = train_keep_prob
        self.use_embedding = use_embedding
        self.embedding_size = embedding_size

        tf.reset_default_graph()
        self.build_inputs()
        self.build_lstm()
        self.build_loss()
        self.build_optimizer()
        self.saver = tf.train.Saver()

    def build_inputs(self):
        #定义placeholder
        with tf.name_scope('inputs'):
            self.inputs = tf.placeholder(tf.int32, shape=(
                self.num_seqs, self.num_steps), name='inputs')
            self.targets = tf.placeholder(tf.int32, shape=(
                self.num_seqs, self.num_steps), name='targets')
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

            # 对于中文，需要使用embedding层
            # 英文字母没有必要用embedding层
            if self.use_embedding is False:
                self.lstm_inputs = tf.one_hot(self.inputs, self.num_classes)
            else:
                with tf.device("/cpu:0"):
                    #输入向量：[num_seqs,num_steps,embedding_size]
                    embedding = tf.get_variable('embedding', [self.num_classes, self.embedding_size])
                    self.lstm_inputs = tf.nn.embedding_lookup(embedding, self.inputs)
                    #下标为input的embedding
                    
    def build_lstm(self):
        # 创建单个cell并堆叠多层
        def get_a_cell(lstm_size, keep_prob):
            lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
            drop = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=keep_prob)
            return drop

        with tf.name_scope('lstm'):
            cell = tf.nn.rnn_cell.MultiRNNCell(
                [get_a_cell(self.lstm_size, self.keep_prob) for _ in range(self.num_layers)]
            )
            self.initial_state = cell.zero_state(self.num_seqs, tf.float32)

            # 通过dynamic_rnn对cell展开时间维度
            self.lstm_outputs, self.final_state = tf.nn.dynamic_rnn(cell, self.lstm_inputs, initial_state=self.initial_state)

            # 通过lstm_outputs得到概率
            #数组连接，按照第2维
            seq_output = tf.concat(self.lstm_outputs, 1)
            x = tf.reshape(seq_output, [-1, self.lstm_size])

            with tf.variable_scope('softmax'):
                softmax_w = tf.Variable(tf.truncated_normal([self.lstm_size, self.num_classes], stddev=0.1))
                softmax_b = tf.Variable(tf.zeros(self.num_classes))

            self.logits = tf.matmul(x, softmax_w) + softmax_b
            self.proba_prediction = tf.nn.softmax(self.logits, name='predictions')

    def build_loss(self):
        with tf.name_scope('loss'):
            y_one_hot = tf.one_hot(self.targets, self.num_classes)
            y_reshaped = tf.reshape(y_one_hot, self.logits.get_shape())
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=y_reshaped)
            self.loss = tf.reduce_mean(loss)

    def build_optimizer(self):
        # 使用clipping gradients
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.grad_clip)
        train_op = tf.train.AdamOptimizer(self.learning_rate)
        self.optimizer = train_op.apply_gradients(zip(grads, tvars))

    def train(self, batch_generator, max_steps, save_path, save_every_n, log_every_n):
        self.session = tf.Session()
        with self.session as sess:
            sess.run(tf.global_variables_initializer())
            # Train network
            step = 0
            new_state = sess.run(self.initial_state)
            for x, y in batch_generator:
                step += 1
                start = time.time()
                feed = {self.inputs: x,
                        self.targets: y,
                        self.keep_prob: self.train_keep_prob,
                        self.initial_state: new_state}
                batch_loss, new_state, _ = sess.run([self.loss,
                                                     self.final_state,
                                                     self.optimizer],
                                                    feed_dict=feed)

                end = time.time()
                # control the print lines
                if step % log_every_n == 0:
                    print('step: {}/{}... '.format(step, max_steps),
                          'loss: {:.4f}... '.format(batch_loss),
                          '{:.4f} sec/batch'.format((end - start)))
                if (step % save_every_n == 0):
                    self.saver.save(sess, os.path.join(save_path, 'model'), global_step=step)
                if step >= max_steps:
                    break
            self.saver.save(sess, os.path.join(save_path, 'model'), global_step=step)

#关于诗词生成，仍然需要进行优化
    def sample(self, n_samples, prime, vocab_size):
        samples = [c for c in prime]
        print (samples)
        sess = self.session
        new_state = sess.run(self.initial_state)
        preds = np.ones((vocab_size, ))  # for prime=[]
        for c in prime:
            x = np.zeros((1, 1))
            # 输入单个字符
            x[0, 0] = c
            feed = {self.inputs: x,
                    self.keep_prob: 1.,
                    self.initial_state: new_state}
            preds, new_state = sess.run([self.proba_prediction, self.final_state],
                                        feed_dict=feed)
        x = np.zeros((1, 1))
            # 输入单个字符
        #矫正

        c=prime[0]
        for i in range(4):
            x = np.zeros((1, 1))
            # 输入单个字符
            x[0, 0] = c
            feed = {self.inputs: x,
                    self.keep_prob: 1.,
                    self.initial_state: new_state}
            preds, new_state = sess.run([self.proba_prediction, self.final_state],
                                        feed_dict=feed)
        #
        c = pick_top_n(preds, vocab_size)
        # 添加字符到samples中
        print(c)
        samples.append(c)
        print(samples)
        print ('第一段samples: {}'.format(np.shape(samples)))
        s=np.array(samples)
        print (converter.arr_to_text(s))

        # 不断生成字符，直到达到指定数目
        for i in range(n_samples):
            x = np.zeros((1, 1))
            x[0, 0] = c
            feed = {self.inputs: x,
                    self.keep_prob: 1.,
                    self.initial_state: new_state}
            preds, new_state = sess.run([self.proba_prediction, self.final_state],
                                        feed_dict=feed)
            
            c = pick_top_n(preds, vocab_size)
            while(c==3500):
                    feed = {self.inputs: x,
                    self.keep_prob: 1.,
                    self.initial_state: new_state}
                    preds, new_state = sess.run([self.proba_prediction, self.final_state],
                                        feed_dict=feed)
            
                    c = pick_top_n(preds, vocab_size)

            samples.append(c)

        print ('第二段samples: {}'.format(np.shape(samples)))
        print (samples)
        #print (converter.int_to_word(samples[5]))
        print (np.array(samples))
        return np.array(samples)
        
    def pline(self, prime,vocab_size,sess,new_state):
        
        n_samples=5
        x=np.zeros((1,1))
        x[0,0]=prime
        #初始化
        sample=[]
        sample.append(prime)
        for i in range(4):
            x = np.zeros((1, 1))
            # 输入单个字符
            feed = {self.inputs: x,
                    self.keep_prob: 1.,
                    self.initial_state: new_state}
            preds, new_state = sess.run([self.proba_prediction, self.final_state],
                                        feed_dict=feed)
        
        for i in range(n_samples):
            feed={self.inputs:x,
                  self.keep_prob:1.,
                  self.initial_state: new_state}
            prebs,new_state = sess.run([self.proba_prediction,self.final_state],
                                       feed_dict=feed)
            c=pick_top_n(prebs,vocab_size)
            while(c==3500):
                    feed = {self.inputs: x,
                    self.keep_prob: 1.,
                    self.initial_state: new_state}
                    preds, new_state = sess.run([self.proba_prediction, self.final_state],
                                        feed_dict=feed)
                    c = pick_top_n(preds, vocab_size)
            
            x=np.zeros((1,1))
            x[0,0]=c
            sample.append(c)
        return sample,new_state
    
    def poemline(self, n_samples, hide_head, vocab_size):
        samples = [hide_head[0]]
        print (samples)
        #初始化
        sess = self.session
        new_state = sess.run(self.initial_state)
        preds = np.ones((vocab_size, ))  # for prime=[]
        #矫正
        start=hide_head[0]
        for i in range(4):
            x = np.zeros((1, 1))
            # 输入单个字符
            x[0, 0] = start
            feed = {self.inputs: x,
                    self.keep_prob: 1.,
                    self.initial_state: new_state}
            preds, new_state = sess.run([self.proba_prediction, self.final_state],
                                        feed_dict=feed)
        #
        c = pick_top_n(preds, vocab_size)
        samples.append(c)
        
        for i in range(4):
            x = np.zeros((1, 1))
            x[0, 0] = c
            feed = {self.inputs: x,
                    self.keep_prob: 1.,
                    self.initial_state: new_state}
            preds, new_state = sess.run([self.proba_prediction, self.final_state],
                                        feed_dict=feed)
            
            c = pick_top_n(preds, vocab_size)
            while(c==3500):
                    feed = {self.inputs: x,
                    self.keep_prob: 1.,
                    self.initial_state: new_state}
                    preds, new_state = sess.run([self.proba_prediction, self.final_state],
                                        feed_dict=feed)
            
                    c = pick_top_n(preds, vocab_size)

            samples.append(c)
        for i in range(3):
            start=hide_head[i+1]
            sample,new_state=self.pline(start,vocab_size,sess,new_state)
            samples.extend(sample)
        print(samples)
        return np.array(samples)
        
    def sample_hide(self, n_samples, head, vocab_size):
        
        samples=[]
        samples.append(head)
        sess = self.session
        new_state = sess.run(self.initial_state)
        preds = np.ones((vocab_size, ))  # for prime=[]
        #矫正
        start=head
        for i in range(4):
            x = np.zeros((1, 1))
            # 输入单个字符
            x[0, 0] = start
            feed = {self.inputs: x,
                    self.keep_prob: 1.,
                    self.initial_state: new_state}
            preds, new_state = sess.run([self.proba_prediction, self.final_state],
                                        feed_dict=feed)
            start = pick_top_n(preds, vocab_size)
        #
                                        
        #c = pick_top_n(preds, vocab_size)
        #samples.append(c)
                                        
        c=head
        for i in range(4):
            x = np.zeros((1, 1))
            x[0, 0] = c
            feed = {self.inputs: x,
                    self.keep_prob: 1.,
                    self.initial_state: new_state}
            preds, new_state = sess.run([self.proba_prediction, self.final_state],
                                        feed_dict=feed)
            
            c = pick_top_n(preds, vocab_size)
            while(c==3500):
                    feed = {self.inputs: x,
                    self.keep_prob: 1.,
                    self.initial_state: new_state}
                    preds, new_state = sess.run([self.proba_prediction, self.final_state],
                                        feed_dict=feed)
            
                    c = pick_top_n(preds, vocab_size)

            samples.append(c)

        print (samples)
        #print (converter.int_to_word(samples[5]))
        print (np.array(samples))
        return samples
    
    def sample_hide_poetry(self, hide_head, vocab_size):
        poetry=[]
        for i in range(4):
            p=self.sample_hide(5,hide_head[i],vocab_size)
            if i%2==0:
                p.append(0)
            else:
                p.append(1)
            poetry.extend(p)
        
        return np.array(poetry)


    def load(self, checkpoint):
        self.session = tf.Session()
        self.saver.restore(self.session, checkpoint)
        print('Restored from: {}'.format(checkpoint))
