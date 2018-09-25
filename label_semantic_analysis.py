"""分析标签之间的语义相似度"""
import word_embedding as we
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

# 标签及其对应种子词
label_seeds = [('very negative', 'bad horrible negative disgusting'),
              ('negative', 'bad confused unexpected useless negative'),
              ('neutral', 'normal moderate neutral objective impersonal'),
              ('positive', 'good positive outstanding satisfied pleased'),
              ('very positive', 'positive impressive unbelievable awesome')]


def get_label_vec(w2id, e, emb_size):
    """获取标签的向量化表示"""
    vec_array = np.zeros([len(label_seeds), emb_size], dtype=np.float32)
    label_list = []
    for index, (label, seeds) in enumerate(label_seeds):
        label_list.append(label)
        vec = np.zeros(emb_size)
        seed_list = seeds.split()
        for seed in seed_list:
            vec += e[w2id[seed]]
        vec /= len(seed_list)
        vec_array[index] = vec
    return vec_array, label_list


def print_sim_matrix(label_list, cos_matrix):
    # 打印相似度矩阵
    out_str = "距离矩阵\t"
    # 打印第一行标签
    for i in range(len(label_list)):
        out_str += label_list[i] + "\t"
    # 换行
    out_str += '\n'
    # 打印混淆矩阵
    for i in range(len(label_list)):
        out_str += label_list[i] + "\t"
        for j in range(len(cos_matrix[i])):
            out_str += str(cos_matrix[i][j]) + '\t'
        out_str += '\n'
    return out_str


def compute_dis_matrix(label_list, vec_array):
    dis_methods = ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan']
    for method in dis_methods:
        dis_matrix = pairwise_distances(vec_array, metric=method)
        print('{}距离:'.format(method))
        print(print_sim_matrix(label_list, dis_matrix))


if __name__=='__main__':
    glove_6B_50d = r'E:\项目\天池比赛\零样本目标检测\glove.6B\glove.6B.50d.txt'
    emb_size = 50
    w2id, id2wt, e = we.load_word2vec(glove_6B_50d, 400000, emb_size)
    vec_array, label_list = get_label_vec(w2id, e, emb_size)
    compute_dis_matrix(label_list, vec_array)
    # cos_matrix = pairwise_distances(vec_array)
    # print_sim_matrix(label_list, cos_matrix)
    pass

