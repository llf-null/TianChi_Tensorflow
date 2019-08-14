# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
tf.set_random_seed(2019)

tf.reset_default_graph()
def plot_decision_boundary(model,x,y):
    #找到x, y的最大值和最小值, 并在周围填充一个像素
    x_min, x_max = x[:, 0].min() -1, x[:, 0].max() +1
    y_min, y_max = x[:, 1].min() -1, x[:, 1].max() +1
    h=0.01
    # 构建一个宽度为`h`的网格
    xx,yy=np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))
    # 计算模型在网格上所有点的输出值
    Z=model(np.c_[xx.ravel(),yy.ravel()])
    Z=Z.reshape(xx.shape)
    #画图显示
    plt.contourf(xx,yy,Z,cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(x[:,0],x[:,1],c=np.squeeze(y),cmap=plt.cm.Spectral)
    
np.random.seed(1)
m=400#样本数
N=int(m/2)#每一类点的个数
D=2#维度
x=np.zeros((m,D))
y=np.zeros((m,1),dtype='uint8') # label 向量，0 表示红色，1 表示蓝色
a=4
for j in range(2):
    ix=range(N*j,N*(j+1))
    t=np.linspace(j*3.12,(j+1)*3.12,N)+np.random.randn(N)*0.2#theta
    r=a*np.sin(4*t)+np.random.randn(N)*0.2#radius
    x[ix]=np.c_[r*np.sin(t),r*np.cos(t)]
    y[ix]=j
plt.scatter(x[:,0],x[:,1],c=np.squeeze(y),s=40,cmap=plt.cm.Spectral)

#1.模型参数需要反复使用，告诉TensorFlow允许复用参数
#tf.get_variable_scope().reuse_variables()

#利用Logistic Regression进行分类
x=tf.constant(x,dtype=tf.float32)
y=tf.constant(y,dtype=tf.float32)

#定义模型
w = tf.get_variable(initializer=tf.random_normal_initializer(), shape=(2, 1),dtype=tf.float32, name='weights')
b = tf.get_variable(initializer=tf.zeros_initializer(), shape=(1), dtype=tf.float32,name='bias')

def logistic_model(x):
    logit=tf.matmul(x,w)+b
    return tf.sigmoid(logit)
y_=logistic_model(x)

#构造训练
loss=tf.losses.log_loss(predictions=y_,labels=y)
lr=1e-1
optimizer=tf.train.GradientDescentOptimizer(learning_rate=lr)
train_op=optimizer.minimize(loss)
#执行训练
sess=tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
for e in range(100):
    sess.run(train_op)
    if (e+1)%200==0:
        loss_numpy = loss.eval(session=sess)
        print('Epoch %d:Loss:%12f'%(e+1,loss_numpy))
#应用到模型上的效果
model_input=tf.placeholder(shape=(None,2),dtype=tf.float32,name='logistic_input')
logistic_output=logistic_model(model_input)
def plot_logistic(x_data):
    y_pred_numpy = sess.run(logistic_output, feed_dict={model_input: x_data})
    out=np.greater(y_pred_numpy,0.5).astype(np.float32)
    return np.squeeze(out)
plot_decision_boundary(plot_logistic,x.eval(session=sess),y.eval(session=sess))




##2.利用深度神经网络进行分类
#
##构建第一个隐藏层
#
##tf.get_variable_scope().reuse_variables()
#with tf.variable_scope('layer1',reuse=None):
#    #构建参数
#    #tf.get_variable_scope().reuse_variables()
#    w1 = tf.get_variable(initializer=tf.random_normal_initializer(stddev=0.01), shape=(2, 4), name='weights1')
#    #tf.get_variable_scope().reuse_variables()
#    b1 = tf.get_variable(initializer=tf.zeros_initializer(), shape=(4), name='bias1')
#
##构建第二个隐藏层
#with tf.variable_scope('layer2',reuse=None):
#    #tf.get_variable_scope().reuse_variables()
#    w2 = tf.get_variable(initializer=tf.random_normal_initializer(stddev=0.01), shape=(4, 1), name='weights2')
#    b2 = tf.get_variable(initializer=tf.zeros_initializer(), shape=(1), name='bias2')
#
##通过上面的参数构建一个两层的神经网络
#def two_network(nn_input):
#    with tf.variable_scope('two_network'):
#        #第一个隐藏层
#        net=tf.matmul(nn_input,w1)+b1
#        #tanh激活函数
#        net=tf.tanh(net)
#        #第二个隐藏层
#        net=tf.matmul(net,w2)+b2
#        #经过sigmoid得出输出
#        return tf.sigmoid(net)
#net=two_network(x)
def hidden_layer(layer_input,output_depth,scope='hidden_layer',reuse=None):
    input_death=layer_input.get_shape()[-1]
    with tf.variable_scope(scope,reuse=reuse):
        w=tf.get_variable(initializer=tf.random_normal_initializer(),shape=(input_death,output_depth),name='weights')
        b=tf.get_variable(initializer=tf.zeros_initializer(),shape=(output_depth),name='bias')
        net_DNN=tf.matmul(layer_input,w)+b
        return net_DNN
def DNN(x,output_depths,scope='DNN',reuse=None):
    net=x
    for i, output_depth in enumerate(output_depths):
        net=hidden_layer(net,output_depth,scope='layer%d'%i,reuse=reuse)
        net=tf.tanh(net)
    net=hidden_layer(net,1,scope='classification',reuse=reuse)
    net=tf.sigmoid(net)
    return net
dnn=DNN(x,[10,10,10])
loss_dnn=tf.losses.log_loss(predictions=dnn,labels=y)
lr=0.1
optimizer=tf.train.GradientDescentOptimizer(learning_rate=lr)
train_op=optimizer.minimize(loss_dnn)

sess.run(tf.global_variables_initializer())
for e in range(50000):
    sess.run(train_op)
    if(e+1)%5000==0:
        loss_numpy=loss_dnn.eval(session=sess)
        print('Epoch{}:Loss:{}'.format(e+1,loss_numpy))
dnn_out=DNN(model_input,[10,10,10],reuse=True)
def plot_dnn(input_data):
    y_pred_numpy=sess.run(dnn_out,feed_dict={model_input:input_data})
    out=np.greater(y_pred_numpy,0.5).astype(np.float32)
    return np.squeeze(out)
plot_decision_boundary(plot_dnn,x.eval(session=sess),y.eval(session=sess))
plt.title('4_layer_network')
        
        
#
##神经网络训练
#loss_two=tf.losses.log_loss(predictions=net,labels=y,scope='loss_two')
#lr=1
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
#train_op=optimizer.minimize(loss=loss_two,var_list=[w1,w2,b1,b2])
#
##模型保存与加载
#saver=tf.train.Saver()
##训练10000次，并在第五千次和最后一次各保存一次模型
#sess.run(tf.global_variables_initializer())
#for e in range(10000):
#    sess.run(train_op)
#    if(e+1)%5000==0:
#        loss_numpy=loss_two.eval(session=sess)
#        print('Epoch{}:Loss:{}'.format(e+1,loss_numpy))
#        #sess参数表示开启模型session，为必选选项
#        #save_path参数表示模型的保存路径，必须以‘ckpt’结尾
#        #global_step参数表示模型当前训练的步数，用来标记不同阶段的模型
#        saver.save(sess=sess,save_path='First_Save/model.ckpt',global_step=(e+1))
#        
##模型效果
#nn_out=two_network(model_input)
#def plot_network(input_data):
#    y_pred_numpy=sess.run(nn_out,feed_dict={model_input:input_data})
#    out=np.greater(y_pred_numpy,0.5).astype(np.float32)
#    return np.squeeze(out)
#plot_decision_boundary(plot_network,x.eval(session=sess),y.eval(session=sess))
#plt.title('2 layer network')
#
#
##模型的恢复(保存在First_Save中的已训练的模型)
#
##恢复模型结构
#saver=tf.train.import_meta_graph('First_Save/model.ckpt-10000.meta')
#
##恢复模型参数
#saver.restore(sess,'First_Save/model.ckpt-10000')
#print(w1.eval(session=sess))
#plot_decision_boundary(plot_network,x.eval(session=sess),y.eval(session=sess))
#plt.title('2 layer network_two')
#








































