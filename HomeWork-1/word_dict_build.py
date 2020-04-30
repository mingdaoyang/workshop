from collections import defaultdict
def save_word_dict(vocab, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        for line in vocab:
            w, i = line
            f.write("%s\t%d\n" % (w, i))


def read_data(path_1, path_2, path_3):
    with open(path_1, 'r', encoding='utf-8') as f1, \
            open(path_2, 'r', encoding='utf-8') as f2, \
            open(path_3, 'r', encoding='utf-8') as f3:
        words = []
        for line in f1:
            words = line.split()

        for line in f2:
            words += line.split(' ')

        for line in f3:
            words += line.split(' ')

    return words


def build_vocab(items, sort=True, min_count=0, lower=False):
    """
    构建词典列表
    :param items: list  [item1, item2, ... ]
    :param sort: 是否按频率排序，否则按items排序
    :param min_count: 词典最小频次
    :param lower: 是否小写
    :return: list: word set
    """
    result = []
    if sort:
        # sort by count
        dic = defaultdict(int) #先生成一个空字典dic，当dic的key不存在时，返回的是工厂函数的默认值，int为0
        for item in items:
            for word in item.split(" "): #使用i很诡异,这里改成word
                word = word.strip()
                if not word: continue #如果word不存在，跳过下两行操作
                word = word if not lower else item.lower()
                dic[word] += 1 #将并统计词频，将word用dic存储，
        # sort
        dic = sorted(dic.items(), key=lambda d: d[1], reverse=True)
        for word, item in enumerate(dic):
            key = item[0]
            if min_count and min_count > item[1]: #当min_count为0时，这个item没出现过，因此跳过下面的操作
                continue
            result.append(key)
    else:
        # sort by items
        for word, item in enumerate(items):
            item = item if not lower else item.lower()
            result.append(item)

    vocab = [(w, i) for i, w in enumerate(result)] #这里的i为经排序枚举后的key
    reverse_vocab = [(i, w) for i, w in enumerate(result)]

    return vocab, reverse_vocab

if __name__ == '__main__':
    lines = read_data('./data/train_x.txt',
                      './data/train_y.txt',
                      './data/test_x.txt')
    vocab, reverse_vocab = build_vocab(lines)
    save_word_dict(vocab, './data/vocab.txt')
