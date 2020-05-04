#按照format={index->vector}构建vocab
from gensim.models import KeyedVectors
import json
from collections import OrderedDict  # discarded in Py3.6 above
from annoy import AnnoyIndex
import time

#定义一个获取运行时间得装饰器函数
def timeit(f):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        res = f(*args, **kwargs)
        end_time = time.time()
        print("函数{}运行时间为{:.6f}s".format(f.__name__, end_time - start_time))
        return res
    return wrapper


# load binary model
def load_model(input_model_path):
     model = KeyedVectors.load_word2vec_format(input_model_path, binary=True)
     return model


def bulit_wv_index(input_model_path):
    # 加载二进制模型
    wv_model = load_model(input_model_path)
    word_index = {}  # 有序字典
    # 给wv_model.vovab中的词加index
    for counter, key in enumerate(wv_model.vocab.keys()):
        word_index[key] = counter
    # 写入json文件
    with open('../data/word_index.json', 'w') as fp:
        json.dump(word_index, fp)
    # 通过Annoy算法建立索引树，256为词向量长度
    wv_index = AnnoyIndex(256, metric='angular') # ?
    i = 0
    for key in wv_model.vocab.keys():
        v = wv_model[key]
        wv_index.add_item(i, v)
        i += 1
    wv_index.build(10)  # 10 trees
    return word_index,wv_index

#生成文件
def save_wv_index_file(wv_index,save_path):
    print('start generate file.......')
    wv_index.save(save_path)
    print('wv_index.index file generation finished')



#get similar words of one keyword
def get_similar_words(word_index,wv_index,kw,similar_word_num):
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    print('和{}类似的前{}个词如下:'.format(kw,similar_word_num))
    for item in wv_index.get_nns_by_item(word_index[kw], similar_word_num):
        print(reverse_word_index[item])


if __name__ == '__main__':

    w2v_path = 'w2v.bin'
    save_path = '../data/word_index.index'
    key_words=['宝马','维修','发动机']
    word_index,wv_index=bulit_wv_index(w2v_path)
    save_wv_index_file(wv_index,save_path)
    for key_word in key_words:
        get_similar_words(word_index,wv_index,key_word,10)

