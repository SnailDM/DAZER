"""零文档过滤模型文件"""
from collections import defaultdict
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

    def test(self):
        """模型预测"""
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config, graph=self.g) as sess:
            train_vars = [v for v in tf.trainable_variables()]
            saver = tf.train.Saver(var_list=train_vars)
            saver.restore(sess, self.ckpt_path + '-77200')  # 是否自动读取最新的一次，还是要手动设置保存的第几次
            test_df = pd.read_csv(self.test_file, engine='python')
            # 打开调试模式
            # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            # 定义结果
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
                    score = sess.run(self.pos_score, feed_dict={self.input_q: query, self.input_pos_d: doc, self.input_q_len: query_len_dict[label]})
                    result[label] += list(np.reshape(score, (-1,)))
            result_df = pd.DataFrame(result)
            result_df.to_csv(self.test_result_file)

    def train(self):
        """模型训练"""
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config, graph=self.g) as sess:
            train_vars = [v for v in tf.trainable_variables()]
            saver = tf.train.Saver(var_list=train_vars, max_to_keep=3)
            sess.run(tf.global_variables_initializer())
            writer = tf.summary.FileWriter(self.summary_path, self.g)
            train_df = pd.read_csv(self.train_file, engine='python')
            step = 0
            # 打开调试模式
            # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            print('训练批次大小:{}'.format(self.batch_size))
            for epoch in range(int(self.max_epochs)):
                if epoch+1 % 25 == 0:
                    sess.run(self.lr.assign(self.model_learning_rate / 5.0))
                # sess.run(self.lr.assign(self.model_learning_rate * 0.95**epoch))
                for batch in dp.input_scale_data_train(train_df, self.train_labels.split(','), batch_size=self.batch_size, min_len=self.max_pooling_num):
                    query = batch['query']
                    label = batch['label']
                    pos_doc = batch['pos_doc']
                    neg_doc = batch['neg_doc']
                    query_len = batch['query_len']
                    merged, ato, mto = sess.run([self.merged, self.adv_train_op, self.model_train_op], feed_dict={self.input_q: query, self.input_l: label, self.input_pos_d: pos_doc, self.input_neg_d: neg_doc, self.input_q_len: query_len})
                    # 可视化
                    writer.add_summary(merged, global_step=step)
                    writer.flush()
                    step += 1
                saver.save(sess, self.ckpt_path, global_step=step)
                print('第{}次epoch训练完成。'.format(epoch+1))
            writer.close()
        pass


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
        dazer.train()
    else:
        dazer.test()


