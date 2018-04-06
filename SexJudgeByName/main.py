#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : main.py
# @Author: Huangqinjian
# @Date  : 2018/4/4
# @Desc  :
import pandas as pd
import tensorflow as tf

# 读取数据
train = pd.read_table('data/train.txt', ',')
test = pd.read_table('data/test.txt', ',')
submit = pd.read_csv('data/sample_submit.csv')

# 所有男生的名字
train_male = train[train['gender'] == 1]

# 所有女生的名字
train_female = train[train['gender'] == 0]

train_x = []
train_y = []

male_len = len(train_male)
female_len = len(train_female)
for i in range(0, male_len):
    train_x.append(train_male.iloc[i, 1])
    train_y.append([0, 1])  # 男
for j in range(0, female_len):
    train_x.append(train_male.iloc[j, 1])
    train_y.append([1, 0])  # 女

max_name_length = max([len(name) for name in train_x])
print("最长名字的字符数: ", max_name_length)
max_name_length = 8

# 词汇表
counter = 0
vocabulary = {}
for name in train_x:
    counter += 1
    tokens = [word for word in name]
    for word in tokens:
        if word in vocabulary:
            vocabulary[word] += 1
        else:
            vocabulary[word] = 1

vocabulary_list = [' '] + sorted(vocabulary, key=vocabulary.get, reverse=True)
print(len(vocabulary_list))

# 字符串转为向量形式
vocab = dict([(x, y) for (y, x) in enumerate(vocabulary_list)])
train_x_vec = []
for name in train_x:
    name_vec = []
    for word in name:
        name_vec.append(vocab.get(word))
    while len(name_vec) < max_name_length:
        name_vec.append(0)
    train_x_vec.append(name_vec)

#######################################################

input_size = max_name_length
num_classes = 2

batch_size = 64
num_batch = len(train_x_vec) // batch_size

X = tf.placeholder(tf.int32, [None, input_size])
Y = tf.placeholder(tf.float32, [None, num_classes])

dropout_keep_prob = tf.placeholder(tf.float32)


def neural_network(vocabulary_size, embedding_size=128, num_filters=128):
    # embedding layer
    with tf.device('/cpu:0'), tf.name_scope("embedding"):
        W = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        # 根据X中的id，寻找W中的对应元素
        embedded_chars = tf.nn.embedding_lookup(W, X)
        # 将维度加1
        embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)
        # convolution + maxpool layer
    filter_sizes = [3, 4, 5]
    pooled_outputs = []
    for i, filter_size in enumerate(filter_sizes):
        with tf.name_scope("conv-maxpool-%s" % filter_size):
            filter_shape = [filter_size, embedding_size, 1, num_filters]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1))
            b = tf.Variable(tf.constant(0.1, shape=[num_filters]))
            conv = tf.nn.conv2d(embedded_chars_expanded, W, strides=[1, 1, 1, 1], padding="VALID")
            h = tf.nn.relu(tf.nn.bias_add(conv, b))
            pooled = tf.nn.max_pool(h, ksize=[1, input_size - filter_size + 1, 1, 1], strides=[1, 1, 1, 1],
                                    padding='VALID')
            # pooled = tf.expand_dims(pooled, 1)
            pooled_outputs.append(pooled)
    # print(pooled_outputs)

    num_filters_total = num_filters * len(filter_sizes)
    # h_pool = tf.concat(3, pooled_outputs)   版本问题
    h_pool = tf.concat(pooled_outputs, 3)
    h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
    # dropout
    with tf.name_scope("dropout"):
        h_drop = tf.nn.dropout(h_pool_flat, dropout_keep_prob)
        # output
    with tf.name_scope("output"):
        W = tf.get_variable("W", shape=[num_filters_total, num_classes],
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.Variable(tf.constant(0.1, shape=[num_classes]))
        output = tf.nn.xw_plus_b(h_drop, W, b)

    return output


# 训练
def train_neural_network():
    output = neural_network(len(vocabulary_list))
    optimizer = tf.train.AdamOptimizer(1e-4)
    # 版本问题
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=Y))
    grads_and_vars = optimizer.compute_gradients(loss)
    train_op = optimizer.apply_gradients(grads_and_vars)

    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for e in range(201):
            for i in range(num_batch):
                batch_x = train_x_vec[i * batch_size: (i + 1) * batch_size]
                batch_y = train_y[i * batch_size: (i + 1) * batch_size]
                _, loss_ = sess.run([train_op, loss], feed_dict={X: batch_x, Y: batch_y, dropout_keep_prob: 0.5})
                print(e, i, loss_)
                # 保存模型
            if e % 50 == 0:
                saver.save(sess, "model/name2sex.model", global_step=e)


# train_neural_network()


# 使用训练的模型
def detect_sex(name_list):
    x = []
    #  得到训练集中每个名字词向量
    for name in name_list:
        name_vec = []
        for word in name:
            if vocab.get(word) is None:
                name_vec.append(0)
            else:
                name_vec.append(vocab.get(word))
        while len(name_vec) < max_name_length:
            name_vec.append(0)
        x.append(name_vec)
    # print(x)

    output = neural_network(len(vocabulary_list))

    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        # 恢复前一次训练
        ckpt = tf.train.get_checkpoint_state('model/')
        if ckpt != None:
            print(ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("没找到模型")

        predictions = tf.argmax(output, 1)
        res = sess.run(predictions, {X: x, dropout_keep_prob: 1.0})
        print(res)

        i = 0
        for name in name_list:
            if res[i] == 0:
                print(name, '女')
            else:
                print(name, '男')
            i += 1


# print(train_male['name'].tolist())
detect_sex(["闳家", "国强", "创海", "方荧", "德章", "钦建", "茂奎", "颛赢", "厅裳", "达其", "歆琪", "鹦应"])
# detect_sex(train_male['name'].tolist())
