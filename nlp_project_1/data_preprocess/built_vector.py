import multiprocessing
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.models.keyedvectors import KeyedVectors


def load_data_from_file(path):
    """
    读文本到数组
    :param path: str 文件路径
    :return:字符串数组 ['line_1','line_2'...]
    """
    print('[load_data_from_file] ... <--- {}'.format(path))
    lines = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            lines.append(line)
    print('[load_data_from_file] FINISHED! data.len:{} '.format(len(lines)))
    return lines


def save_data_to_file(data, path):
    """
    写数据到文本文件
    :param data:字符串数组
    :param path:文件路径
    """
    print('[save_data_to_file] ... data.len={}'.format(len(data)))
    with open(path, 'w', encoding='utf-8') as f:
        for line in data:
            f.write('{}\n'.format(line))
    print('[save_data_to_file] FINISHED! ---> {}'.format(path))


def build_word2vec(sentens_path, w2v_path, w2v_bin_path, min_count=10, window=5, size=256, sg=1, iter=5):
    print('[build_word2vec] STARTED...')
    print('Train Model STARTED...')
    w2v = Word2Vec(sg=sg, sentences=LineSentence(sentens_path), size=size, window=window, min_count=min_count,
                   workers=multiprocessing.cpu_count(), iter=iter)

    print('Train Model FINISHED! \n Save Model STARTED!')
    w2v.wv.save_word2vec_format(w2v_path, binary=False)
    w2v.wv.save_word2vec_format(w2v_bin_path, binary=True)
    print('Save Model FINISHED!...')


def model_test(model, kw_1, kw_2):
    print('--------------------------------------------------------------------------------------')
    print('{} vs {} similarity ：{}'.format(kw_1, kw_2, model.similarity(kw_1, kw_2)))
    print('--------------------------------------------------------------------------------------')
    print('{} similar verbs contains：{}'.format(kw_1, model.similar_by_word(kw_1)))
    print('{} similar verbs contains：{}'.format(kw_2, model.similar_by_word(kw_2)))

if __name__ == '__main__':
    train_x_cut_file_path = '../data/train_x.txt'
    train_y_cut_file_path = '../data/train_y.txt'
    test_x_cut_file_path = '../data/test_x.txt'

    # gen
    all_cut_lines_file_path = '../data/all_cut_file_path.txt'
    word2vec_file_path = '../data/word2vec.txt'
    word2vec_bin_file_path = '../data/word2vec_bin.txt'

    all_cut_lines = []
    all_cut_lines += load_data_from_file(train_x_cut_file_path)
    all_cut_lines += load_data_from_file(train_y_cut_file_path)
    all_cut_lines += load_data_from_file(test_x_cut_file_path)

    save_data_to_file(all_cut_lines, all_cut_lines_file_path)

    build_word2vec(all_cut_lines_file_path, word2vec_file_path, word2vec_bin_file_path,
                   min_count=100, window=5, size=256, sg=1, iter=5)

    # tests :
    w2v_model = KeyedVectors.load_word2vec_format(word2vec_bin_file_path, binary=True)
    model_test(w2v_model, '宝马', '奔驰')
    model_test(w2v_model, '汽车', '减速')
    model_test(w2v_model, '汽车', '车子')
    model_test(w2v_model, '技师', '车主')
    model_test(w2v_model, '火花塞', '减震器')
    model_test(w2v_model, '刹车片', '解答')
