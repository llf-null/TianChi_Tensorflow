# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
#1.构造图
    #1.1简答加减
a=tf.constant(32)
b=tf.constant(10)
c=tf.add(a,b)
    #1.2矩阵运算
mat_a=tf.constant([1,2,3,4])
mat_a=tf.reshape(mat_a,[2,2])
mat_b=tf.constant([1,2,3,4,5,6])
mat_b=tf.reshape(mat_b,[2,3])
mat_c=tf.matmul(mat_a,mat_b)
    #1.3随机化
rand_normal=tf.random_normal((),mean=0.0,stddev=1.0,dtype=tf.float32,seed=None)
#2执行图
sess=tf.Session()
print(sess.run(c))
print(sess.run(mat_c))
print(sess.run(rand_normal))
#3关闭session
sess.close()

#2.variable(定义图)
var_a=tf.Variable(3,dtype=tf.int32)
var_b=tf.Variable(4,dtype=tf.int32)
    #2.1variable初始化(之后还是要sess.run)
sess=tf.InteractiveSession()
        #2.1.1一次性初始化所有变量
init=tf.global_variables_initializer()
init.run()
        #2.1.2初始化某些变量
#init_ab=tf.variables_initializer([var_a,var_b])
#init_ab.run()
        #2.1.3初始化某个变量
#var_a.initializer.run()
    #2.2使用sess执行图(也可以使用函数自身调用eval来执行图)
#print(sess.run([var_a,var_b]))
print(var_a.eval())
    #2.3variable赋值(要将赋值操作也定义为op去初始化他才可以赋值成功)
assign_op=var_a.assign(100)
var_a.initializer.run()
assign_op.eval()
print(var_a.eval())
sess.close()

#3占位符placeholder
a=tf.placeholder(tf.float32,shape=[3])
    #3.1执行图
sess=tf.Session()
print(sess.run(a,feed_dict={a:[1,2,3]}))

#4graph
#g=tf.get_default_graph
#for op in g.get_operations():
#    print(op.name)
#5.graph可视化
with tf.Session()as sess:
    graph_writer=tf.summary.FileWriter('.',sess.graph)

