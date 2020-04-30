from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.models.word2vec import keep_vocab_item
'''
class gensim.models.word2vec.LineSentence(source, max_sentence_length=10000, limit=None)
source：通常是文件的路径或某个打开的文件对象
对包含句子的文件进行迭代:一行=一句话；单词必须经过预处理，并用空格分隔。
'''
from gensim.models.keyedvectors import KeyedVectors #实现了词向量及其相似性查找
import pickle  #实现了数据序列和反序列化（二进制）
import os  #os模块提供了非常丰富的方法用来处理文件和目录
import numpy as np


def readlines(path):
    lines = []
    with open(path,'r',encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            lines.append(line)
    return  lines


def dump_pkl(vocab,pkl_path,overwrite=True):
    if pkl_path and os.path.exists(pkl_path) and not overwrite:
        return
    if pkl_path:
        with open(pkl_path,'wb') as f: #"wb" 以二进制写方式打开，只能写文件
            pickle.dump(vocab,f,protocol=pickle.HIGHEST_PROTOCOL)
            # pickle.dump(obj, file, [,protocol])
            # 接受一个文件句柄和一个数据对象作为參数，把数据对象obj以特定的格式保存到给定的文件file里
            #-----------------------------------------------------------------------
            #protocol表示生成的二进制文件和不同的python版本之间的兼容性
            #例如，协议版本1是旧的二进制格式，它也与Python的早期版本兼容

        print('save %s ok' % pkl_path)

def extract_sentences(train_x_path,train_y_path,test_x_path):#合并
    ret = []
    lines = readlines(train_x_path)
    lines += readlines(train_y_path)
    lines += readlines(test_x_path)
    for line in lines:
        ret.append(line)
    return  ret

def save_sentence(lines,sentence_path):
    with open(sentence_path,'w',encoding='utf-8') as f:
        for line in  lines:
            f.write('%s\n' %  line.strip())
    f.close()

def build(train_x_path,train_y_path,test_x_path,out_path=None,sentence_path='',
         w2v_bin_path='w2v.bin',min_count=100 ):
    sentences = extract_sentences(train_x_path,train_y_path,test_x_path)
    save_sentence(sentences,sentence_path)
    w2v = Word2Vec(sentences=LineSentence(sentence_path),size=256,window=5,min_count=200,iter=5)
    w2v.wv.save_word2vec_format(w2v_bin_path,binary=True)
    model = KeyedVectors.load_word2vec_format(w2v_bin_path,binary=True)


if __name__ == '__main__':

    build(train_x_path='./data/train_x.txt',
          train_y_path='./data/train_y.txt',
          test_x_path='./data/test_x.txt',
          out_path='./data/word2vec.txt',
          sentence_path='./data/sentences.txt')
