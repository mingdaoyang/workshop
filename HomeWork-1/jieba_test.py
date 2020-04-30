import numpy as np
import pandas as pd
import re
#上面import的包在下面的代码中并未用到，会显示灰色，不用大惊小怪！
import jieba
from jieba import posseg #jieba.posseg可以对词性进行标注，然而在本项目中并没什么卵用
#定义一个专用分词函数，以方便后续调用
def segment(sentence,cut_type='word',pos=False):
    """
    :param sentence:           后面要把数据集拼接为句子
    :param cut_type:           粒度：word和char级别
    :param pos:                是否标注词性，默认不标注
    :return:
    """
    if pos:#整体上从是否支持词性标注进行判断
        if cut_type == 'word':
            word_pos_seq = posseg.lcut(sentence)
            '''
            注意cut和lcut的区别：
            cut是一个生成器(generator),需要通过for循环取其中的每一个词
            而lcut直接生成一个list
           '''
            word_seq,pos_seq = [],[]
            #定义两个list，第一个用于存放所分出的词，第二个用于存放其词性（如果pos=Ture）
            for w,p in word_pos_seq:
                word_seq.append(w)
                pos_seq.append(p)
            return word_seq,pos_seq
        elif cut_type == 'char':
            word_seq = list(sentence)
            pos_seq = []
            for w in word_seq:
                w_p = posseg.lcut(w)
                #print(w_p)的输出如下：和'word'不大一样，生成的是一个pair
                '''
                [pair('我', 'r')]
                [pair('比', 'p')]
[
               '''
                pos_seq.append(w_p[0].flag)  #这里取pair中0号位置的flag(词性)
            return word_seq,pos_seq

    else:
        if cut_type == 'word':
            return jieba.lcut(sentence)
        elif cut_type == 'char':#由于粒度是char，可以直接用list返回
            return list(sentence)

#测试一下效果
print(segment('我比较喜欢武汉的生活','word'))
print(segment('我比较喜欢武汉的生活','char'))
print(segment('我比较喜欢武汉的生活','word',True))
print(segment('我比较喜欢武汉的生活','char',True))