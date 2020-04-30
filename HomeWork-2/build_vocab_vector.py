from gensim.models import KeyedVectors
import json
from collections import OrderedDict
from annoy import AnnoyIndex
from annoy import AnnoyIndex
def built_vocab(input_modle_path):
    #加载二进制模型
    wv_model = KeyedVectors.load_word2vec_format(input_modle_path, binary=True)
    word_index = OrderedDict() #有序字典
    #给wv_model.vovab中的词加index
    for counter, key in enumerate(wv_model.vocab.keys()):
        word_index[key] = counter
    #写入json文件
    with open('word_index.json', 'w') as fp:
        json.dump(word_index, fp)
    #通过Annoy算法建立索引树，256为词向量长度
    wv_index = AnnoyIndex(256,metric='angular')
    i = 0
    #针对wv_index中的每一个index i 写入相应的词向量,和原始dic的index不同
    for word in wv_model.vocab.keys():
        v = wv_model[word]
        wv_index.add_item(i, v)
        i += 1
    wv_index.build(10) #10 trees
    wv_index.save('wv_index.txt')

if __name__=='__main__':
    path = 'w2v.bin'
    built_vocab(path)
