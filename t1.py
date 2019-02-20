# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# import numpy as np
# import matplotlib.pyplot as plt
# import xlrd
# import utils
import tensorflow as tf

# 第一步必须先初始化一个graph变量，然后在这个graph里进行图计算
graph = tf.Graph()
with graph.as_default():
    variable = tf.Variable(42, name='foo')
    # 初始化所有的变量
    initialize = tf.global_variables_initializer()
    assign = variable.assign(13)

    matrix1 = tf.constant([[3., 3.]])
    matrix2 = tf.constant([[2.], [2.]])
    product = tf.matmul(matrix1, matrix2)
with tf.Session(graph=graph) as sess:
    # sess.run(initialize)
    # sess.run(assign)
    # print(sess.run(variable))

    print(sess.run(matrix1))
    print(sess.run(matrix2))
    print(sess.run(product))

    sess.close()

with tf.Session() as sess:
    with tf.device('/cpu:0'):
        matrix1 = tf.constant([[3., 3.]])
        matrix2 = tf.constant([[2.], [2.]])
        product = tf.matmul(matrix1, matrix2)
        print(sess.run(product))

        sess.close()