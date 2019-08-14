# -*- coding: utf-8 -*-
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
#导入数据
x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],                    [9.779], [6.182], [7.59], [2.167], [7.042],                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)
y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],                    [3.366], [2.596], [2.53], [1.221], [2.827],                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)
#plt.plot(x_train,y_train,'bo')
x=tf.constant(x_train,name='x')
y=tf.constant(y_train,name='y')
#定义参数和模型
w=tf.Variable(initial_value=tf.random_normal(shape=(),seed=2019),dtype=tf.float32,name='weight')
b=tf.Variable(initial_value=0,dtype=tf.float32,name='biase')
with tf.variable_scope('Linear_Model'):
    y_pred=w*x+b
#print(w.name)
#print(y_pred.name)
#最开始的估测参数散点图
sess=tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
#y_pred_numpy=y_pred.eval(session=sess)
#plt.plot(x_train,y_train,'bo',color='r',label='real')
#plt.plot(x_train,y_pred_numpy,'bo',label='estimated')
#plt.legend()
#plt.show()
#定义误差函数
loss=tf.reduce_mean(tf.square(y-y_pred))
print(loss.eval(session=sess))
#gradient descent
w_grad,b_grad=tf.gradients(loss,[w,b])
print('w_grad:',w_grad.eval(session=sess))
print('b_grad:',b_grad.eval(session=sess))
#learing rate
lr=1e-2
w_update=w.assign_sub(lr*w_grad)
b_update=b.assign_sub(lr*b_grad)
for e in range(3):
    sess.run([w_update,b_update])
y_pred_numpy=y_pred.eval(session=sess)
plt.plot(x_train,y_train,'bo',color='r',label='real')
plt.plot(x_train,y_pred_numpy,'bo',label='estimated')
plt.legend()
plt.show()
