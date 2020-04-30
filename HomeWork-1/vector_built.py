from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
def train_w2v_model(path):
    w2v_model = Word2Vec(LineSentence(path),workers=4,size=50,min_count=1)
    w2v_model.save('w2v.model')
    #w2v_model.wv.save('new_name') #以二进制保存，速度较快
if __name__ == '__main__':
    path1 = '.\data\small_corpus.txt'
    path2 = '.\data\sentences.txt'
    train_w2v_model(path1)
    model = Word2Vec.load('w2v.model')
    print(model.wv.most_similar('车'))
    #可以使用LineSentence简化以下代码
    # with open(path2,'r',encoding='utf-8') as f:
    #    data = f.readlines() #读取所有行，注意和readline的区别
    #    f.close()
    new_words = []
    for line in LineSentence(path2): #使用LineSentence更简洁
        new_words.append(line)
    model.train(sentences=new_words,epochs=1,total_examples=len(new_words)) #total_examples参数为行数
    print(model.wv.most_similar('发动机'))