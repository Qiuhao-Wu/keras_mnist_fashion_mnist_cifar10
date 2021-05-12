# -*- coding: utf-8 -*-
import tensorflow as tf

# 配置神经网络的参数
INPUT_NODE = 784  # 输入层的节点数。图像像素为28*28=784
OUTPUT_NODE = 10  # 输出层的节点数。等于类别的数目，0~9共10个数。

# 定义与样本数据相关的参数
IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10

# 第一层卷积层的尺寸和深度
CONV1_DEEP = 32
CONV1_SIZE = 5

# 第二层卷积层的尺寸和深度
CONV2_DEEP = 64
CONV2_SIZE = 5

# 全连接层的节点个数
FC_SIZE = 512


def inference(input_tensor, train, regularizer):
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable("weight",
                                        [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP],
                                        # 参数一、二：过滤器得尺寸；参数三：当前层得深度；参数四：过滤器的深度
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))

        conv1_biases = tf.get_variable("bias",
                                       [CONV1_DEEP],
                                       initializer=tf.constant_initializer(0.0))

        conv1 = tf.nn.conv2d(input_tensor,
                             conv1_weights,
                             strides=[1, 1, 1, 1],  # 1、4位置固定为1；2、3位为不同维度上的步长，通常为1
                             padding='SAME')  # SAME表示添加0填充；VALID表示不添加
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    # 实现第二层池化层的前向传播过程。这里选用最大池化层，池化层过滤器的边长为2，
    # 使用全0填充且移动的步长为2。这一层的输入是上一层的输出，也就是28x28x32 #的矩阵。输出为14x14x32的矩阵。
    with tf.name_scope('layer2-pool1'):
        pool1 = tf.nn.max_pool(relu1,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')

    # 声明第三层卷积层的变量并实现前向传播过程。这一层的输入为14x14x32的矩阵。输出为14x14x64的矩阵。
    # 通过tf.get_variable的方式创建过滤器的权重变量和偏置项变量。
    with tf.variable_scope('layer3-conv2'):
        conv2_weights = tf.get_variable("weight",
                                        [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("bias",
                                       [CONV2_DEEP],
                                       initializer=tf.constant_initializer(0.0))

        conv2 = tf.nn.conv2d(pool1,
                             conv2_weights,
                             strides=[1, 1, 1, 1],
                             padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    with tf.name_scope('layer4-pool2'):
        pool2 = tf.nn.max_pool(relu2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')

    pool_shape = pool2.get_shape().as_list()  #pool2: [batch_size,7,7,64]
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3] #7*7*64
    # 通过tf.reshape函数将第四层的输出编程一个batch的向量
    reshaped = tf.reshape(pool2, [pool_shape[0], nodes])

    # 声明第五层全连接层的变量并实现前向传播过程。这一层的输入是拉直之后的一组向量，
    # 向量长度为7*7*64=3136，输出是一组长度为512的向量。此处引入了 dropout的概念。
    # dropout在训练时会随机将部分节点的输出改为0。dropout可以避免过拟合问题，从而使得模型在测试数据上的效果更好。
    # dropout一般只在全连接层而不是卷积层或者池化层使用。
    with tf.variable_scope('layer5-fc1'):
        fc1_weights = tf.get_variable("weight",
                                      [nodes, FC_SIZE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))

        # 只有全连接层的权重需要加入正则
        # 当给出了正则化生成函数时，将当前变量的正则化损失加入名字为losses的集合。
        # 在这里使用了add_to_collection函数将一个张量加入一个集合，而这个集合的名称为losses。
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc1_weights))

        fc1_biases = tf.get_variable("bias",
                                     [FC_SIZE],
                                     initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
        if train: fc1 = tf.nn.dropout(fc1, 0.5)

    # 声明第六层全连接层的变量并实现前向传播过程。这一层的输入为一组长度为512的向量，
    # 输出为一组长度为10的向量。这一层的输出通过Softmax之后就得到了最后的分类结果。
    with tf.variable_scope('layer6-fc2'):
        fc2_weights = tf.get_variable("weight",
                                      [FC_SIZE, NUM_LABELS],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))

        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc2_weights))

        fc2_biases = tf.get_variable("bias",
                                     [NUM_LABELS],
                                     initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(fc1, fc2_weights) + fc2_biases

    return logit

