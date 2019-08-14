# -*- coding: utf-8 -*-
import tensorflow as tf
def layer(input, weight_shape, bias_shape):
# 变量初始化器， 将作为get_variable的参数
    weight_init = tf.random_uniform_initializer(minval=-1,maxval=1)
    print('1')
    bias_init = tf.constant_initializer(value=0)
    print('2')
# 使用initializer指定上面创建的初始化器
    W = tf.get_variable("W", weight_shape,initializer=weight_init)
    print('3')
    b = tf.get_variable("b", bias_shape,initializer=bias_init)
    print('4')
    return tf.matmul(input, W) + b

def my_network(input):
        # 为每一层网路创建一个特有的域
    with tf.variable_scope("layer_1"):
        output_1 = layer(input, [784, 100], [100])
    with tf.variable_scope("layer_2"):
        output_2 = layer(output_1, [100, 50], [50])
    with tf.variable_scope("layer_3"):
        output_3 = layer(output_2, [50, 10], [10])
    return output_3
with tf.variable_scope("shared_variables",reuse=True) as scope:
    i_1 = tf.placeholder(tf.float32, [1000, 784], name="i_1")
    my_network(i_1)

    i_2 = tf.placeholder(tf.float32, [1000, 784],name="i_2")
    my_network(i_2)