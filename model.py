"""零文档过滤模型文件"""
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics
from collections import defaultdict
from Logger import Logger
import logging
import os
from tensorflow.python import debug as tf_debug
import argparse
import data_process as dp
import pandas as pd
import numpy as np
import word_embedding as we
import tensorflow as tf
from traitlets.config.loader import PyFileConfigLoader
from traitlets.config import Configurable
from traitlets import (
    Int,
    Float,
    Bool,
    Unicode,
)

# 获取当前目录
_get_abs_path = lambda path: os.path.normpath(os.path.join(os.getcwd(), path))
# 初始化日志类
Logger(logname=os.path.join(_get_abs_path('log'), 'dazer.log'), loglevel=1, logger="main")
logger = logging.getLogger('main')



def dd1_list():
    return defaultdict(list)


def weight_init(shape, name):
    # tmp = 1.0 / np.sqrt(shape[0])
    tmp = np.sqrt(3.0) / np.sqrt(shape[0] + shape[1])  # 论文用到的初始化值
    return tf.Variable(initial_value=tf.random_uniform(shape, minval=-tmp, maxval=tmp), name=name)


def l2_loss(var_list):
    """计算一组参数的l2损失"""
    loss = 0.0
    for var in var_list:
        loss += tf.nn.l2_loss(var)  # 计算单个参数的l2损失
    return loss


class Dazer(Configurable):
    emb_file = Unicode('None', help='词向量文件路径').tag(config=True)
    vocabulary_size = Int(400000, help='加载的词汇数量').tag(config=True)
    embedding_size = Int(50, help='词潜入的纬度').tag(config=True)
    train_file = Unicode('None', help='训练样本文件').tag(config=True)
    train_labels = Unicode('None', help='需要训练的分类标签').tag(config=True)
    all_labels = Unicode('None', help='全分类标签').tag(config=True)

    test_file = Unicode('None', help='训练样本文件').tag(config=True)
    ckpt_path = Unicode('None', help='模型保存路径').tag(config=True)
    test_result_file = Unicode('None', help='测试数据结果').tag(config=True)
    summary_path = Unicode('None', help='统计标量保存路径').tag(config=True)

    kernal_num = Int(50, help="滤波器个数").tag(config=True)
    kernal_width = Int(5, help="滤波器宽度").tag(config=True)

    max_epochs = Int(10, help="最大训练轮数").tag(config=True)
    eval_frequency = Int(1, help="将数据写入可视化的频率").tag(config=True)
    batch_size = Int(16, help="批次大小").tag(config=True)
    load_model = Bool(False, help="是否加载已经训练的模型").tag(config=True)

    max_pooling_num = Int(3, help="最池化k值").tag(config=True)
    decoder_mlp1_num = Int(75, help="解码第一层全连接的神经元个数").tag(config=True)

    regular_term = Float(0.01, help='正则项系数，对抗分类和相似性回归 共享这个超参数').tag(config=True)
    adv_learning_rate = Float(0.001, help='对抗分类的学习率').tag(config=True)
    epsilon = Float(0.00001, help='对抗分类的学习率').tag(config=True)
    model_learning_rate = Float(0.001, help='相似性模型的学习率').tag(config=True)
    adv_loss_term = Float(0.2, help='adv损失在总损失中占的比例').tag(config=True)

    """零样本过滤模型"""
    def __init__(self, **kwargs):
        super(Dazer, self).__init__(**kwargs)
        # 定义类的实例属性
        self.word2id, self.id2word, self.emb = we.load_word2vec(self.emb_file, self.vocabulary_size,
                                                                self.embedding_size)
        self.class_num = len(self.train_labels.split(','))
        # 初始化数据流图
        self.g = tf.Graph()
        self.structure_init()

    def structure_init(self):
        """定义数据流图结构"""
        with self.g.as_default():
            # 定义输入
            self.input_q = tf.placeholder(dtype=tf.int32, shape=[None, None])
            self.input_q_len = tf.placeholder(dtype=tf.float32, shape=[None, ])
            self.input_pos_d = tf.placeholder(dtype=tf.int32, shape=[None, None])
            self.input_neg_d = tf.placeholder(dtype=tf.int32, shape=[None, None])
            self.input_l = tf.placeholder(dtype=tf.int32, shape=[None, self.class_num])

            # 词嵌入
            emb_q = tf.nn.embedding_lookup(self.emb, self.input_q)
            emb_pos_d = tf.nn.embedding_lookup(self.emb, self.input_pos_d)
            emb_neg_d = tf.nn.embedding_lookup(self.emb, self.input_neg_d)

            # 生成标签的门向量
            class_vec = tf.divide(tf.reduce_sum(emb_q, axis=1), tf.expand_dims(self.input_q_len, axis=-1))
            query_gate_weights = weight_init([self.embedding_size, self.kernal_num], 'gate_weights')
            query_gate_bias = tf.Variable(initial_value=tf.zeros(self.kernal_num, ),  name='gate_bias')
            # shape:[bath_size, kernal_num]
            gate_vec = tf.sigmoid(tf.matmul(class_vec, query_gate_weights) + query_gate_bias)
            rs_gate_vec = tf.expand_dims(gate_vec, axis=1)


            # 生成文档向量
            pos_sub_info = tf.subtract(tf.expand_dims(class_vec, axis=1), emb_pos_d)
            pos_mul_info = tf.multiply(emb_pos_d, tf.expand_dims(class_vec, axis=1))
            conv_pos_input = tf.expand_dims(tf.concat([emb_pos_d, pos_sub_info, pos_mul_info], -1), axis=-1)

            neg_sub_info = tf.subtract(tf.expand_dims(class_vec, axis=1), emb_neg_d)
            neg_mul_info = tf.multiply(emb_neg_d, tf.expand_dims(class_vec, axis=1))
            conv_neg_input = tf.expand_dims(tf.concat([emb_neg_d, neg_sub_info, neg_mul_info], -1), axis=-1)

            # 卷积操作提取文档窗口特征
            pos_conv = tf.layers.conv2d(inputs=conv_pos_input, filters=self.kernal_num,
                                        kernel_size=(self.kernal_width, self.embedding_size*3),
                                        strides=(1, self.embedding_size*3), padding="same",
                                        name='doc_conv', trainable=True)
            neg_conv = tf.layers.conv2d(inputs=conv_neg_input, filters=self.kernal_num,
                                        kernel_size=(self.kernal_width, self.embedding_size * 3),
                                        strides=(1, self.embedding_size * 3), padding="same",
                                        name='doc_conv', trainable=True, reuse=True)
            # shape=[batch,max_dlen,1,kernal_num]
            # reshape to [batch,max_dlen,kernal_num]
            rs_pos_conv = tf.squeeze(pos_conv, [2])
            rs_neg_conv = tf.squeeze(neg_conv, [2])

            pos_gate_conv = tf.multiply(rs_gate_vec, rs_pos_conv)
            neg_gate_conv = tf.multiply(rs_gate_vec, rs_neg_conv)

            # top_k 池化
            transpose_pos_gate_conv = tf.transpose(pos_gate_conv, [0, 2, 1])  # reshape to [batch,kernal_num,max_dlen]
            pos_conv_k_max_pooling, _ = tf.nn.top_k(transpose_pos_gate_conv, self.max_pooling_num)
            pos_encoder = tf.reshape(pos_conv_k_max_pooling, [-1, self.kernal_num * self.max_pooling_num])

            transpose_neg_gate_conv = tf.transpose(neg_gate_conv, [0, 2, 1])
            neg_conv_k_max_pooling, _ = tf.nn.top_k(transpose_neg_gate_conv, self.max_pooling_num)
            neg_encoder = tf.reshape(neg_conv_k_max_pooling, [-1, self.kernal_num * self.max_pooling_num])

            pos_decoder_mlp1 = tf.layers.dense(inputs=pos_encoder, units=self.decoder_mlp1_num, activation=tf.nn.tanh,
                                               trainable=True, name='decoder_mlp1')
            neg_encoder_mlp1 = tf.layers.dense(inputs=neg_encoder, units=self.decoder_mlp1_num, activation=tf.nn.tanh,
                                               trainable=True, name='decoder_mlp1', reuse=True)

            self.pos_score = tf.layers.dense(inputs=pos_decoder_mlp1, units=1, activation=tf.nn.tanh,
                                             trainable=True, name='decoder_mlp2')
            neg_score = tf.layers.dense(inputs=neg_encoder_mlp1, units=1, activation=tf.nn.tanh,
                                        trainable=True, name='decoder_mlp2', reuse=True)
            hinge_loss = tf.reduce_mean(tf.maximum(0.0, 1 - self.pos_score + neg_score))
            tf.summary.scalar('hinge_loss', hinge_loss)

            # 对抗学习
            adv_weight = weight_init([self.decoder_mlp1_num, self.class_num], 'adv_weights')
            adv_bias = tf.Variable(initial_value=tf.zeros(self.class_num, ), name='adv_bias')

            adv_prob = tf.nn.softmax(tf.add(tf.matmul(pos_decoder_mlp1, adv_weight), adv_bias))
            adv_prob_log = tf.log(adv_prob)
            adv_loss = tf.reduce_mean(tf.reduce_sum(tf.multiply(adv_prob_log, tf.cast(self.input_l, tf.float32)), axis=1))
            adv_l2_loss = self.regular_term * l2_loss([v for v in tf.trainable_variables() if 'b' not in v.name and 'adv' in v.name])
            loss_cat = -1 * adv_loss + adv_l2_loss
            # 动态学习率
            self.lr = tf.Variable(self.model_learning_rate, trainable=False, name='learning_rate')
            self.adv_train_op = tf.train.AdamOptimizer(learning_rate=self.lr, epsilon=self.epsilon)\
                .minimize(loss_cat, var_list=[v for v in tf.trainable_variables() if 'adv' in v.name])
            tf.summary.scalar('loss_cat', loss_cat)

            model_l2_loss = self.regular_term*l2_loss([v for v in tf.trainable_variables() if 'b' not in v.name and 'adv' not in v.name])
            loss = hinge_loss + model_l2_loss + adv_loss*self.adv_loss_term
            self.model_train_op = tf.train.AdamOptimizer(learning_rate=self.lr, epsilon=self.epsilon)\
                .minimize(loss, var_list=[v for v in tf.trainable_variables() if 'adv' not in v.name])
            tf.summary.scalar('loss', loss)
            self.merged = tf.summary.merge_all()

    def analysis_result(self, sess, test_df):
        """分析预测结果"""
        result = dd1_list()
        for batch in dp.input_scale_data_test(test_df, self.max_pooling_num, batch_size=self.batch_size):
            query_dict = batch['query_dict']
            query_len_dict = batch['query_len_dict']
            doc = batch['doc']
            l_context = batch['l_context']
            l_label = batch['l_label']
            # 生成结果字典
            result['real_label'] += l_label
            result['context'] += l_context
            for label, query in query_dict.items():
                score = sess.run(self.pos_score, feed_dict={self.input_q: query, self.input_pos_d: doc,
                                                            self.input_q_len: query_len_dict[label]})
                result[label] += list(np.reshape(score, (-1,)))
        result_df = pd.DataFrame(result)
        # 自动获取标签
        labels = [label for label, query in query_dict.items()]
        # 写死标签顺序， 其他数据集需要修改此标签顺序
        # labels = ['very negative', 'negative', 'neutral', 'positive', 'very positive']

        result_df['pre_label'] = result_df[labels].idxmax(axis=1)
        f1 = metrics.f1_score(result_df['real_label'], result_df['pre_label'], average='weighted')
        result_report = classification_report(result_df['real_label'], result_df['pre_label'])
        con_matrix = confusion_matrix(result_df['real_label'], result_df['pre_label'], labels=labels)
        return result_df, result_report, f1, labels, con_matrix

    def test(self):
        """模型预测"""
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config, graph=self.g) as sess:
            train_vars = [v for v in tf.trainable_variables()]
            saver = tf.train.Saver(var_list=train_vars)
            saver.restore(sess, self.ckpt_path + '-77212')  # 是否自动读取最新的一次，还是要手动设置保存的第几次
            test_df = pd.read_csv(self.test_file, engine='python')
            # 打开调试模式
            # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            # 定义结果
            result_df, result_report, f1, labels, con_matrix = self.analysis_result(sess, test_df)
            logger.info('生成测试集预测统计，并保存具体预测结果：')
            logger.info('\n'+result_report)
            result_df.to_csv(self.test_result_file)

    # def train(self):
    #     """单独一次零样本模型训练"""
    #     config = tf.ConfigProto()
    #     config.gpu_options.allow_growth = True
    #     with tf.Session(config=config, graph=self.g) as sess:
    #         train_vars = [v for v in tf.trainable_variables()]
    #         saver = tf.train.Saver(var_list=train_vars, max_to_keep=3)
    #         sess.run(tf.global_variables_initializer())
    #         writer = tf.summary.FileWriter(self.summary_path, self.g)
    #         train_df = pd.read_csv(self.train_file, engine='python')
    #         test_df = pd.read_csv(self.test_file, engine='python')
    #         step = 0
    #         # 打开调试模式
    #         # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    #         logger.info('训练批次大小:{}'.format(self.batch_size))
    #         for epoch in range(int(self.max_epochs)):
    #             if epoch+1 % 25 == 0:
    #                 sess.run(self.lr.assign(self.model_learning_rate / 5.0))
    #             # sess.run(self.lr.assign(self.model_learning_rate * 0.95**epoch))
    #             for batch in dp.input_scale_data_train(train_df, self.train_labels.split(','), batch_size=self.batch_size, min_len=self.max_pooling_num):
    #                 query = batch['query']
    #                 label = batch['label']
    #                 pos_doc = batch['pos_doc']
    #                 neg_doc = batch['neg_doc']
    #                 query_len = batch['query_len']
    #                 merged, ato, mto = sess.run([self.merged, self.adv_train_op, self.model_train_op], feed_dict={self.input_q: query, self.input_l: label, self.input_pos_d: pos_doc, self.input_neg_d: neg_doc, self.input_q_len: query_len})
    #                 # 可视化
    #                 writer.add_summary(merged, global_step=step)
    #                 writer.flush()
    #                 step += 1
    #             # 每5次做一次验证并打印
    #             if epoch % 5 == 0:
    #                 logger.info('第{}次训练结果：'.format(epoch+1))
    #                 logger.info('训练样本和零样本的准确率：')
    #                 self.analysis_result(sess, train_df)
    #                 logger.info('测试样本和零样本的准确率：')
    #                 self.analysis_result(sess, test_df)
    #             saver.save(sess, self.ckpt_path, global_step=step)
    #             logger.info('第{}次epoch训练完成。'.format(epoch+1))
    #         writer.close()
    #     pass

    def train_all(self):
        """自动循环训练所有标签，使得每一个标签都做为零样本标签，并观察其泛化效果"""
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        now_time = datetime.now().strftime('%Y%m%d%H%M')
        train_df = pd.read_csv(self.train_file, engine='python')
        test_df = pd.read_csv(self.test_file, engine='python')
        logger.info('------------------------------------------------------------------------------------------------')
        logger.info('启动全样本训练, 时间:{}'.format(now_time))

        for zero_label, train_labels, train_data, test_data in reorganize_data(self.all_labels, train_df, test_df):
            #  单独对negative零样本标签，进行调参
            if zero_label != 'very positive':
                continue
            else:
                # self.regular_term = 0.0005  # negative 正则项系数
                # self.regular_term = 0.0001  # very negative 正则项系数
                self.regular_term = 0.001  # positive ， very positive 正则项系数
                self.adv_learning_rate = 0.01  # positive ， very positive 学习率

            with tf.Session(config=config, graph=self.g) as sess:
                train_vars = [v for v in tf.trainable_variables()]
                saver = tf.train.Saver(var_list=train_vars, max_to_keep=3)
                sess.run(tf.global_variables_initializer())
                max_f1 = 0.0
                best_test_result_report = ''
                best_con_matrix_report = ''

                step = 0
                # 保存在特定标签下路径中
                writer = tf.summary.FileWriter(self.summary_path.format(now_time, zero_label.replace(' ', '')), self.g)
                logger.info('开启学习---零样本标签:{}'.format(zero_label))
                logger.info(
                    '本次训练embedding_size：{}. max_epochs：{}, batch_size: {}, regular_term：{}, adv_learning_rate:'
                    '{}, epsilon：{}, adv_loss_term：{}'.format(self.embedding_size, self.max_epochs, self.batch_size
                                                              , self.regular_term, self.adv_learning_rate,
                                                              self.epsilon, self.adv_loss_term))
                # 动态学习率
                for epoch in range(int(self.max_epochs)):
                    # 动态学习率
                    # if epoch + 1 % 25 == 0:
                    #     sess.run(self.lr.assign(self.model_learning_rate / 5.0))  # 倍数缩小设置方式
                    #     sess.run(self.lr.assign(self.model_learning_rate * 0.95**epoch))  # 指数设置方式
                    for batch in dp.input_scale_data_train(train_data, train_labels,
                                                           batch_size=self.batch_size, min_len=self.max_pooling_num):
                        query = batch['query']
                        label = batch['label']
                        pos_doc = batch['pos_doc']
                        neg_doc = batch['neg_doc']
                        query_len = batch['query_len']
                        merged, ato, mto = sess.run([self.merged, self.adv_train_op, self.model_train_op],
                                                    feed_dict={self.input_q: query, self.input_l: label,
                                                               self.input_pos_d: pos_doc, self.input_neg_d: neg_doc,
                                                               self.input_q_len: query_len})
                        # 可视化
                        writer.add_summary(merged, global_step=step)
                        writer.flush()
                        step += 1
                    # 每5次做一次验证并打印
                    if epoch % 5 == 0:
                        logger.info('第{}次训练结果：'.format(epoch + 1))
                        logger.info('训练样本的准确率：')
                        train_result_df, train_result_report, train_f1, train_labels, train_con_matrix = self.analysis_result(sess, train_data)
                        logger.info('\n'+train_result_report)
                        logger.info('测试样本和零样本的准确率：')
                        test_result_df, test_result_report, test_f1, test_labels, test_con_matrix = self.analysis_result(sess, test_data)
                        logger.info('\n'+test_result_report)
                        logger.info('测试样本和零样本的混淆矩阵：{}'.format(test_labels))
                        test_con_matrix_report = con_matrix_2_str(test_labels, test_con_matrix)
                        logger.info('\n' + test_con_matrix_report)
                        if test_f1 > max_f1:
                            # 保存权重值
                            saver.save(sess, self.ckpt_path.format(now_time, zero_label.replace(' ', '')), global_step=step)
                            # 记录最好结果的统计报告 和 混淆矩阵
                            best_test_result_report = test_result_report
                            best_con_matrix_report = test_con_matrix_report
                            # 保存预测的结果
                            test_result_df.to_csv(self.test_result_file.format(now_time, zero_label.replace(' ', '')))
                            max_f1 = test_f1
                    if epoch + 1 == int(self.max_epochs):
                        logger.info('测试集上最好的结果：')
                        logger.info('\n'+best_test_result_report)
                        logger.info('\n' + best_con_matrix_report)
                logger.info('第{}次epoch训练完成。'.format(epoch + 1))
                writer.close()


def reorganize_data(all_labels, train_df, test_df):
    """重新组织训练数据"""
    all_labels_list = all_labels.split(',')
    for label in all_labels_list:
        zero_labels = set([label])
        train_labels = set(all_labels_list) - zero_labels
        # 将零样本标签排除在训练集之外
        train_data = train_df[True ^ train_df['label'].isin(zero_labels)]
        # 将零样本标签数据和测试集合并
        test_data = pd.concat([train_df[train_df['label'].isin(zero_labels)], test_df])
        yield label, list(train_labels), train_data, test_data


def con_matrix_2_str(labels, con_matrix):
    out_str = "混淆矩阵\t"
    # 打印第一行标签
    for i in range(len(labels)):
        out_str += labels[i] + "\t"
    # 换行
    out_str += '\n'
    # 打印混淆矩阵
    for i in range(len(con_matrix)):
        out_str += labels[i] + "\t"
        for j in range(len(con_matrix[i])):
            out_str += str(con_matrix[i][j]) + '\t'
        out_str += '\n'
    return out_str


if __name__ == '__main__':
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='命令行选项描述.')
    parser.add_argument("--config", '-c', type=str, default='sample.config')
    # 添加互斥组，用来区分命令行是训练，还是用来做预测
    group = parser.add_argument_group()
    group.add_argument("--train", action='store_true')

    group.add_argument("--test", action='store_true')
    args = parser.parse_args()
    conf = PyFileConfigLoader(args.config).load_config()
    # 训练模型
    dazer = Dazer(config=conf)
    if args.train:
        dazer.train_all()
    else:
        dazer.test()


