# -*- coding=utf-8 -*-
import tensorflow as tf
from read_utils import TextConverter
from model import CharRNN
import os
import codecs
from IPython import embed

FLAGS = tf.flags.FLAGS


tf.flags.DEFINE_integer('lstm_size', 128, 'size of hidden state of lstm')
tf.flags.DEFINE_integer('num_layers', 2, 'number of lstm layers')
tf.flags.DEFINE_boolean('use_embedding', True, 'whether to use embedding')
tf.flags.DEFINE_integer('embedding_size', 128, 'size of embedding')
tf.flags.DEFINE_string('converter_path', 'app/poem/model/poetry/converter.pkl', 'model/name/converter.pkl')
tf.flags.DEFINE_string('checkpoint_path', 'app/poem/model/poetry/', 'checkpoint path')
tf.flags.DEFINE_string('start_string', u'春夏秋冬', 'use this string to start generating')
tf.flags.DEFINE_integer('max_length', 101, 'max length to generate')

lstm_size=128
num_layers=2
use_embedding=True
embedding_size=128
converter_path='app/poem/model/poetry/converter.pkl'
checkpoint_path='app/poem/model/poetry/'
start_string=u'春夏秋冬'
max_length=101

def poem_genetate(poem_start=u'君'):
    #FLAGS.start_string = FLAGS.start_string
    #FLAGS.start_string = FLAGS.start_string.decode('utf-8')
    converter = TextConverter(filename=FLAGS.converter_path)
    if os.path.isdir(FLAGS.checkpoint_path):
        FLAGS.checkpoint_path =tf.train.latest_checkpoint(FLAGS.checkpoint_path)
        print FLAGS.checkpoint_path
    """
    model = CharRNN(converter.vocab_size, sampling=True,
                    lstm_size=FLAGS.lstm_size, num_layers=FLAGS.num_layers,
                    use_embedding=FLAGS.use_embedding,
                    embedding_size=FLAGS.embedding_size)
                    """
    model = CharRNN(converter.vocab_size, sampling=True,
                    lstm_size=lstm_size, num_layers=num_layers,
                    use_embedding=use_embedding,embedding_size=FLAGS.embedding_size)

    model.load(FLAGS.checkpoint_path)

    #start = converter.text_to_arr(start_string)
    start1 = converter.text_to_arr(poem_start)
    arr = model.sample(max_length, start1, converter.vocab_size)
    #pl = model.poemline(max_length, start, converter.vocab_size)
    #sp=model.sample_hide_poetry( start, converter.vocab_size)
    poem=converter.arr_to_text(arr)
    #print (converter.arr_to_text(sp))
    print('---------')
    print(poem)
    print('---------')
    #print(converter.arr_to_text(pl))
    print('---------')
    #0:, 1:。 2:\n,每行12个字符。不可以有0,1,2大于1个
    
    lines=poem.split('\n')
    r_poem=[]
    for i in range(len(lines)):
        if len(lines[i])==12:
            count=0
            print lines[i][5]
            if lines[i][5]==',':
                print "true"
            if lines[i][5]==u'，':
                print "u true"
            if lines[i][5]==u'，' and lines[i][11]==u'。':
                for j in range(len(lines[i])):
                    if lines[i][j]==u'，' or lines[i][j]==u'。':
                        count+=1
                if count==2:
                    r_poem.append(lines[i])
        if len(r_poem)==2:
            break

    """
    lines=poem.split('\n')
    r_poem=[]
    for i in range(len(lines)):
        if len(lines[i])==12:
            count=0
            if lines[i][5]==0 and lines[i][11]==1:
                for j in range(len(lines[i])):
                    if lines[i][j]==0 or lines[i][j]==1:
                        count+=1
                if count==2:
                    r_poem.append(lines[i])
        if len(r_poem)==2:
            break
            """
    with codecs.open("app/poem.txt","w",'utf-8') as f:
        words="".join(r_poem)
        print (lines)
        print (r_poem)
        print (words)
    
        #words=words.decode('utf-8')
        f.write(words)
        
def poem_interface(poem_start):
    poem_genetate(poem_start=poem_start)
    
if __name__ == '__main__':
    poem_genetate()
