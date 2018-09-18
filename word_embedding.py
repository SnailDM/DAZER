"""词嵌入，可以加载不同类型的词向量"""
import numpy as np


# 加载预训练的词向量
def load_word2vec(file, vocabulary_size, embedding_size):
    # 单词转id字典
    word2id_dict = {}
    # id转单词数组
    id2word_list = []
    # 先创建好固定大小0矩阵，然后再修改值，效率比较高.其中第一行为空白填充：#p#a#d#
    emb = np.zeros([vocabulary_size + 1, embedding_size], dtype=np.float32)
    # 添加空白符的嵌入向量[0,0...]
    word2id_dict['#p#a#d#'] = 0
    id2word_list.append('#p#a#d#')
    # 加载其他单词的嵌入
    with open(file, encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i+1 > vocabulary_size:
                break
            item = line.split()
            word2id_dict[item[0]] = i+1
            id2word_list.append(item[0])
            emb[i+1, :] = [float(value) for value in item[1:]]

    return word2id_dict, id2word_list, emb


if __name__ == '__main__':
    glove_6B_50d = r'E:\项目\天池比赛\零样本目标检测\glove.6B\glove.6B.50d.txt'
    w2id, id2wt, e = load_word2vec(glove_6B_50d, 400000, 50)
    pass



