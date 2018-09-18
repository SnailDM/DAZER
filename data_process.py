"""数据预处理，代码需要根据数据源格式的不同进行调整"""

import tensorflow
import os
import pandas as pd
import random
import numpy as np
import word_embedding as we


def input_20news_train(path, types=None):
    """20类新闻的输入"""
    pass


def input_scale_data_train(train_df, labels,  min_len, batch_size=16):
    """读取整合后的电影评论数据"""
    labels_reverse_index = get_reverse_index(labels)
    # 筛选出选中标签的训练数据
    train_df_labels = train_df[train_df['label'].isin(labels)]
    # 定义负采样的索引序列, 并随机打乱
    neg_start_index = 0
    neg_index_list = list(range(len(train_df_labels)))
    random.shuffle(neg_index_list)
    # 定义正采样的索引序列，并随机打乱
    pos_index_list = list(range(len(train_df_labels)))
    random.shuffle(pos_index_list)
    # 定义输出结构
    l_q = []  # 种子词
    l_q_len = []  # 种子词长度
    l_l = []  # 样本所属标签，one hot形式
    l_d = []  # 样本内容
    l_d_aux = []  # 负样本内容
    # 输出数据
    for index in pos_index_list:
        row = train_df_labels.iloc[index]
        # 提取文档的单词编码
        q = np.array([int(word) for word in row['swords_id'].split()])
        l = one_hot(row['label'], labels_reverse_index)
        d = np.array([int(word) for word in row['subj_id'].split()])
        neg_start_index, adv_row = adv_sample(neg_start_index, neg_index_list, row['label'], train_df_labels)
        d_aux = np.array([int(word) for word in adv_row['subj_id'].split()])

        # 加入批样本中
        l_q.append(q)
        l_q_len.append(len(q))
        l_l.append(l)
        l_d.append(d)
        l_d_aux.append(d_aux)

        # 当达到样本批次数量时
        if len(l_q) >= batch_size:
            # 用0填充l_q， l_d， l_d_aux使得每行的长度一致，符合numpy对于矩阵的定义
            query = pad_space(l_q)
            query_len = np.array(l_q_len, dtype=np.int32)
            label = np.array(l_l, dtype=np.int)
            pos_doc = pad_space(l_d, min_len)
            neg_doc = pad_space(l_d_aux, min_len)
            batch = {'query': query, 'label': label, 'pos_doc': pos_doc, 'neg_doc': neg_doc, 'query_len': query_len}
            yield batch
            l_q, l_q_len, l_l, l_d, l_d_aux = [], [], [], [], []
    # if len(l_q) > 0:
    #     query = pad_space(l_q)
    #     label = np.array(l_l, dtype=np.int)
    #     pos_doc = pad_space(l_d)
    #     neg_doc = pad_space(l_d_aux)
    #     batch = {'query': query, 'label': label, 'pos_doc': pos_doc, 'neg_doc': neg_doc}
    #     yield batch


def extend_arr(value, shape, dtype=np.int32):
    """按规定形状，以最后维度扩展value"""
    mat = np.ones(shape, dtype=dtype)
    return mat * value


def input_scale_data_test(test_df, min_len, batch_size=16):
    """给出测试模型效果的批量样本"""
    # 抽取存在标签及其对应种子词id
    label_dict = {row['label']: [int(word) for word in row['swords_id'].split()] for index, row in test_df.drop_duplicates(subset=['label'], keep='first').iterrows()}
    query_dict = {label: extend_arr(swords_id, (batch_size, len(swords_id))) for label, swords_id in
                  label_dict.items()}
    query_len_dict = {label: extend_arr(len(swords_id), (batch_size,)) for label, swords_id in
                      label_dict.items()}
    # 定义输出结构
    l_label = []
    l_context = []
    l_d = []  # 样本内容
    for index, row in test_df.iterrows():
        d = np.array([int(word) for word in row['subj_id'].split()])
        l_d.append(d)
        l_label.append(row['label'])
        l_context.append(row['subj'])
        if len(l_d) >= batch_size:
            doc = pad_space(l_d, min_len)
            batch = {'query_dict': query_dict, 'doc': doc, 'query_len_dict': query_len_dict, 'l_label': l_label, 'l_context': l_context}
            yield batch
            l_d, l_label, l_context = [],  [],  []
    if len(l_d) > 0:
        doc = pad_space(l_d, min_len)
        query_dict = {label: extend_arr(swords_id, (len(l_d), len(swords_id))) for label, swords_id in
                      label_dict.items()}
        query_len_dict = {label: extend_arr(len(swords_id), (len(l_d),)) for label, swords_id in
                          label_dict.items()}
        batch = {'query_dict': query_dict, 'doc': doc, 'query_len_dict': query_len_dict, 'l_label': l_label, 'l_context': l_context}
        yield batch


def pad_space(l, min_len=0, data_type=np.int):
    """
    填充l每行中的空白
    :param l:
    :param padding:
    :param data_type:
    :return: numpy数组
    """
    l_len = len(l)
    max_row_len = max(max([len(row) for row in l]), min_len)
    l_pad = np.zeros([l_len, max_row_len], dtype=data_type)
    for i in range(l_len):
        l_pad[i][0: len(l[i])] = l[i]
    return l_pad


def adv_sample(neg_start_index, neg_index_list, current_label, train_df_labels):
    """
    负样本采样
    :param neg_start_index: 负样本索引号
    :param neg_index_list: 负样本索引列表
    :param current_label: 当前样本的标签
    :param train_df_labels: 数据
    :return:负样本
    """
    while True:
        # 按顺序从打乱后的索引序列中取出某行
        adv_row = train_df_labels.iloc[neg_index_list[neg_start_index % len(neg_index_list)]]
        neg_start_index += 1
        if adv_row['label'] != current_label:
            return neg_start_index, adv_row


def one_hot(label, labels_reverse_index):
    """将一个标签编码为一个向量"""
    vec = np.zeros(len(labels_reverse_index), dtype=np.int)
    vec[labels_reverse_index[label]] = 1
    return vec


def get_reverse_index(labels):
    """
    获取标签的索引号，用于one hot 编码
    :param labels: 标签数组
    :return: 倒排索引
    """
    reverse_index = {}
    for index, label in enumerate(labels):
        reverse_index[label] = index
    return reverse_index


def scale_data_process(path, out_path, word2id):
    """电影评论数据的整合"""
    rating_list = []
    subj_list = []
    for folder in os.listdir(path):
        folder_path = os.path.join(path, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                if 'rating' in file:
                    rating_list += read_file_2_list(file_path)
                elif 'subj' in file:
                    subj_list += read_file_2_list(file_path)
                else:
                    continue
    rating_list = [float(rating) for rating in rating_list]
    label_list, seed_words_list = classify_rule(rating_list)
    # 单词的id化------以后可以尝试词形还原、小写归一化、去停用词、去标点符号，用来验证清洗对效果的影响
    seed_words_id_list = [' '.join([str(word2id[word]) for word in words.split() if word in word2id]) for words in seed_words_list]
    subj_id_list = [' '.join([str(word2id[word]) for word in words.split() if word in word2id]) for words in subj_list]
    norm_df = pd.DataFrame({'subj': subj_list, 'rating': rating_list, 'label': label_list, 'swords': seed_words_list, 'subj_id': subj_id_list, 'swords_id': seed_words_id_list})
    # 随机分层抽样20%作为测试集
    train_df, test_df = stratify_sample(norm_df, 'label', 0.8)
    train_df.to_csv(os.path.join(out_path, 'scale_data_train.csv'))
    test_df.to_csv(os.path.join(out_path, 'scale_data_test.csv'))


def stratify_sample(df, label_col, scale):
    """分层抽样，输入一个dataframe的label列和比例"""
    labels = df[label_col].unique()
    train_index_list = []
    test_index_list = []
    # 对每一类进行抽样
    for label in labels:
        index_list = df[df[label_col] == label].index.tolist()
        positive_index, negative_index = random_sample(index_list, scale)
        train_index_list += positive_index
        test_index_list += negative_index
    positive_df = df.iloc[train_index_list]
    negative_df = df.iloc[test_index_list]
    return positive_df, negative_df


def random_sample(index_list, scale):
    """随机抽样，输入一个数组和抽样比例，则返回两个抽样后的结果"""
    positive_index_list = []
    negative_index_list = []
    lenth = len(index_list)
    if index_list is not None and lenth > 0:
        positive_index_list = random.sample(index_list, int(lenth*scale))
        negative_index_list = list(set(index_list) - set(positive_index_list))
    return positive_index_list, negative_index_list


def classify_rule(rating_list):
    """
    根据分值，对标签进行分类
    :param rating_list: 分值数组
    :return: 对应的分类数组和种子词数组
    """
    label_list, seed_words_list = [], []
    for rating in rating_list:
        """分类规则"""
        # 极负面
        if 0.0 <= rating <= 0.2:
            label_list.append('very negative')
            seed_words_list.append('bad horrible negative disgusting ')
        # 负面
        elif 0.2 < rating <= 0.4:
            label_list.append('negative')
            seed_words_list.append('bad confused unexpected useless negative')
        # 中性
        elif 0.4 < rating <= 0.6:
            label_list.append('neutral')
            seed_words_list.append('normal moderate neutral objective impersonal')
        # 正面
        elif 0.6 < rating <= 0.8:
            label_list.append('positive')
            seed_words_list.append('good positive outstanding satisfied pleased')
        # 极正面
        else:
            label_list.append('very positive')
            seed_words_list.append('positive impressive unbelievable awesome')
    return label_list, seed_words_list


def read_file_2_list(file):
    """读取文件"""
    with open(file) as f:
        li = [line.rstrip('\n') for line in f]
    return li


if __name__=="__main__":
    # emb_file = r'E:\项目\天池比赛\零样本目标检测\glove.6B\glove.6B.50d.txt'
    # vocabulary_size = 400000
    # embedding_size = 50
    # word2id, id2word, emb = we.load_word2vec(emb_file, vocabulary_size, embedding_size)
    # scaledata_ori_path = r'E:\项目\天池比赛\零样本目标检测\data\scale_data\scaledata'
    # scaledata_out_path = r'E:\项目\天池比赛\零样本目标检测\data\scale_data\norm_data'
    # scale_data_process(scaledata_ori_path, scaledata_out_path, word2id)

    test_df = pd.read_csv(r'E:\项目\天池比赛\零样本目标检测\data\scale_data\norm_data\scale_data_test.csv', engine='python')
    a = list(input_scale_data_test(test_df, 3, batch_size=16))
    pass

