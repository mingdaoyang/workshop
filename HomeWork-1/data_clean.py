import numpy as np
import pandas as pd
import re
#上面import的包在下面的代码中如果暂未用到，会变灰，不用大惊小怪！
import jieba
from jieba import posseg #jieba.posseg可以对词性进行标注，然而在本项目中并没什么卵用

#定义一些可从数据中能够明显观察到的需要删除的符号和词
REMOVE_WORDS = ['|', '[', ']', '语音', '图片']

def remove_words(words_list):
    #带if的列表式for
    words_list = [word for word in words_list if word not in REMOVE_WORDS]
    return words_list

#读取停用词
def read_stopwords(path):
    lines = set() #创建一个无序不重复元素集，初始为空，注意这个set函数
    with open(path, mode='r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()#去除换行符
            lines.add(line) #set().add() 添加数据
    return lines

#分词函数
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
               '''
                pos_seq.append(w_p[0].flag)  #这里取pair中0号位置的flag(词性)
            return word_seq,pos_seq

    else:
        if cut_type == 'word':
            return jieba.lcut(sentence)
        elif cut_type == 'char':#由于粒度是char，可以直接用list返回
            return list(sentence)


def parse_data(train_path,test_path):
    #首先读取训练数据
    train_df = pd.read_csv(train_path,encoding='utf-8')
    #去除report为空的样本
    train_df.dropna(subset=['Report'],how='any',inplace=True)
    '''
    :param axis:        default：0（跨行操作） 1（跨列操作）；      
    :param how:         {‘any’, ‘all’}, default ‘any’指带缺失值的所有行;'all’指清除全是缺失值的行；
    :param thresh:      int,保留含有int个非空值的行；   
    :param subset:      对特定的列进行缺失值删除处理；          
    :param inplace:     这个很常见,True表示就地更改，视频里讲师说有这个参数不需要对结果进行赋值。
    '''
    #填充x为空的样本
    train_df.fillna('',inplace = True)
    train_x = train_df.Question.str.cat(train_df.Dialogue)
    train_y = train_df.Report
    assert  len(train_x) == len(train_y)
    #针对test数据进行上面同样的操作
    test_df = pd.read_csv(test_path,encoding='utf-8')
    test_df.fillna('',inplace=True)
    test_x = test_df.Question.str.cat(test_df.Dialogue)
    test_y = [] #标签为空，待训练生成
    return train_x,train_y,test_x,test_y
#保存预处理后的数据
def save_data(data_1,data_2,data_3,data_path_1,data_path_2,data_path_3,stop_words_path):
    stopwords = read_stopwords(stop_words_path)
    # 以下为针对train_x的处理
    with open(data_path_1,'w',encoding='utf-8') as f1:
        #count_1 = 0
        for line in data_1:
            if isinstance(line,str):#做一下数据类型检查
                seg_list = segment(line.strip(),cut_type='word')#先分词，再处理，这里的line.strip()去除换行符
                seg_list = remove_words(seg_list)#去除一些特殊符号
                seg_list=[word for word in seg_list if word not in stopwords]#用了一个带if的for列表式循环
                if len(seg_list) != 0:
                    seg_line = ' '.join(seg_list) #分词之间加一个空格
                    f1.write('%s' % seg_line)
                    f1.write('\n')
                else:
                    print('有空值')
                #count_1 += 1
        #print('train x length is ',count_1)

        # 以下为针对train_y的处理
        with open(data_path_2, 'w', encoding='utf-8') as f2:
            #count_2 = 0
            for line in data_2:
                if isinstance(line, str):  # 做一下数据类型检查
                    seg_list = segment(line.strip(), cut_type='word')  # 调用前面定义的jieba分词函数，这里的line.strip()去除换行符
                    # 去除一些特殊符号
                    seg_list = remove_words(seg_list)
                    seg_list = [word for word in seg_list if word not in stopwords]  #用了一个带if的for循环列表式
                    if len(seg_list) != 0:
                        seg_line = ' '.join(seg_list)  # 分词之间加一个空格
                        f2.write('%s' % seg_line)
                        f2.write('\n')
                    else:
                        f2.write('随时 联系')
                        f2.write('\n')
                    #count_2 += 1
            #print('train_y length is ', count_2)

            # 以下为针对test_x的处理
            with open(data_path_3, 'w', encoding='utf-8') as f3:
                #count_3 = 0
                for line in data_3:
                    if isinstance(line, str):  # 做一下数据类型检查
                        seg_list = segment(line.strip(), cut_type='word')  # 调用前面定义的jieba分词函数，这里的line.strip()去除换行符
                        # 去除一些特殊符号
                        seg_list = remove_words(seg_list)
                        seg_list = [word for word in seg_list if word not in stopwords]  # 用了一个带if的for循环列表式
                        if len(seg_list) != 0:
                            seg_line = ' '.join(seg_list)  # 分词之间加一个空格
                            f3.write('%s' % seg_line)
                            f3.write('\n')#注意写入后要换行
                        else:
                            print('有空值')
                        #count_3 += 1
                #print('test_x length is ', count_3)

if __name__ =='__main__':

    train_x,train_y,test_x,_ = parse_data('./data/AutoMaster_TrainSet.csv','./data/AutoMaster_TestSet.csv')
    print("train_x->%s" % len(train_x))
    print("train_y->%s" % len(train_y))
    print(" test_x->%s" % len(test_x))
    save_data(train_x,train_y,test_x,'./data/train_x.txt','./data/train_y.txt','./data/test_x.txt',stop_words_path = './data/stop_words.txt')