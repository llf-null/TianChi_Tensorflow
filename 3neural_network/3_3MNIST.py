# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.examples.tutorials.mnist.input_data as input_data
tf.set_random_seed(2019)
mnist=input_data.read_data_sets('MNIST_data',one_hot=True)
tf.reset_default_graph()

train_set=mnist.train
test_set=mnist.test

fig,axes=plt.subplots(ncols=6,nrows=2)
plt.tight_layout(w_pad=-2.0,h_pad=-8.0)
# 调用next_batch方法来一次性获取12个样本,这里有一个`shuffle`参数, 表达是否打乱样本间的顺序
images,labels=train_set.next_batch(12,shuffle=False)
for ind,(image,label) in enumerate(zip(images,labels)):
    # image 是一个 784 维的向量, 是图片进行拉伸产生的, 这里我们给它 reshape 回去
    image=image.reshape((28,28))
    # label 是一个 10 维的向量, 哪个下标处的值为1 说明是数字几
    label=label.argmax()
    row=ind//6
    col=ind%6
    axes[row][col].imshow(image,cmap='gray')#灰度图
    axes[row][col].axis('off')
    axes[row][col].set_title('%d'%label)

##定义深度网络结构
#def hidden_layer(layer_input, output_depth, scope='hidden_layer', reuse=None):
#    input_depth = layer_input.get_shape()[-1]
#    with tf.variable_scope(scope, reuse=reuse):
#        #初始化方法使用truncated_normal
#        w = tf.get_variable(initializer=tf.truncated_normal_initializer(stddev=0.1),shape=(input_depth, output_depth), name='weights')
#        #使用0.1初始化bias
#        b = tf.get_variable(initializer=tf.constant_initializer(0.1), shape=(output_depth), name='bias')
#        net = tf.matmul(layer_input, w) +b
#        return net
#def DNN(x, output_depths, scope='DNN', reuse=None):
#    net=x
#    for i,output_depth in enumerate(output_depths):
#        net = hidden_layer(net, output_depth, scope='layer%d'%i, reuse=reuse)
#        #激活函数使用relu
#        net = tf.nn.relu(net)
#    #因为数字是十分类问题，因此对应于one_hot标签输出是一个10维向量
#    net = hidden_layer(net, 10, scope='classification', reuse=reuse)
#    return net
#
##定义占位符
#input_ph = tf.placeholder(shape=(None, 784), dtype=tf.float32)
#label_ph = tf.placeholder(shape=(None, 10), dtype=tf.int64)
#
##构造四层神经网络，隐藏节点分别为400,200,100,10
#dnn = DNN(input_ph, [400, 200, 100])
#
##定义loss function，选用交叉熵函数
#loss = tf.losses.softmax_cross_entropy(logits=dnn, onehot_labels=label_ph)
##定义正确率
#acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(dnn, axis=-1), tf.argmax(label_ph,axis=-1)), dtype=tf.float32))
#lr=0.01
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
#train_op = optimizer.minimize(loss)
#
#sess = tf.InteractiveSession()
#
##循环训练20000次
#batch_size=64
#sess.run(tf.global_variables_initializer())
#for e in range(2000):
#    #获取batch_size个样本
#    images, labels = train_set.next_batch(batch_size)
#    sess.run(train_op, feed_dict={input_ph: images, label_ph: labels})
#    if e%100==99:
#        #获取batch_size 个样本
#        test_imgs, test_labels = test_set.next_batch(batch_size)
#        #计算当前样本上的训练以及测试样本的loss和正确率
#        loss_train, acc_train = sess.run([loss, acc], feed_dict={input_ph: images,label_ph: labels})
#        loss_test, acc_test = sess.run([loss, acc], feed_dict={input_ph: test_imgs,label_ph: test_labels})
#        print('STEP {}: train_loss: {:.6f} train_acc: {:.6f} test_loss: {:.6f}test_acc: {:.6f}'.format(e+1, loss_train, acc_train, loss_test, acc_test))
#print('Train Done')
#print('-'*30)
#
##计算所有训练样本的损失值和正确率
#train_loss=[]
#train_acc=[]
#for _ in range(train_set.num_examples//100):
#    image,label=train_set.next_batch(100)
#    loss_train,acc_train=sess.run([loss,acc],feed_dict={input_ph:image,label_ph:label})
#    train_loss.append(loss_train)
#    train_acc.append(acc_train)
#print('Train data loss:{:.6f}'.format(np.array(train_loss).mean()))
#print('Train data acc:{:.6f}'.format(np.array(train_acc).mean()))
#
##计算所有测试样本的损失值和正确率
#test_loss=[]
#test_acc=[]
#for _ in range(test_set.num_examples//100):
#    image,label=test_set.next_batch(100)
#    loss_test,acc_test=sess.run([loss,acc],feed_dict={input_ph:image,label_ph:label})
#    test_loss.append(loss_test)
#    test_acc.append(acc_test)
#print('Test data loss:{:.6f}'.format(np.array(test_loss).mean()))
#print('Test data acc:{:.6f}'.format(np.array(test_acc).mean()))
#sess.close()

#可视化训练
#显示误差变化
loss_sum=tf.summary.scalar('loss',loss)
#显示权重w变化
w_hist=tf.summary.scalar('w_hist',w)
##显示灰度图（如一通道灰度图，3通道RGB，4通道RGBA的变化）
#image_sum = tf.summary.image('image', image)
##显示三维、二维变化
#audio_sum=tf.summary.audio('audio',audio)
#重置计算图
tf.reset_default_graph()
#重新定义占位符
input_ph=tf.placeholder(shape=(None,784),dtype=tf.float32)
label_ph=tf.placeholder(shape=(None,10),dtype=tf.int64)
#重构前向神经网络，构造隐藏层和参数的函数内部构造tf.summary
#构造权重，使用tf.truncated_normal初始化
def weight_variable(shape):
    init=tf.truncated_normal(shape=shape,stddev=0.1)
    return tf.Variable(init)
#构造bias，使用0.1初始化
def bias_variable(shape):
    init=tf.constant(0.1,shape=shape)
    return tf.Variable(init)
#构造添加variable的summary函数
def variable_summaries(var):
    with tf.name_scope('summaries'):
        #计算平均值
        mean=tf.reduce_mean(var)
        #将平均值添加到summary中，mean是数，用scalar表示
        tf.summary.scalar('mean',mean)
        #计算标准差
        with tf.name_scope('stddev'):
            stddev=tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        #将标准差添加到summary中
        tf.summary.scalar('stddev',stddev)
        #添加最大值、最小值到summary
        tf.summary.scalar('max',tf.reduce_max(var))
        tf.summary.scalar('min',tf.reduce_min(var))
        #添加变量分布情况到summary，用histogram
        tf.summary.histogram('histogram',var)
#构建一个hidden layer
def hidden_layer(x,output_dim,scope='hidden_layer',act=tf.nn.relu,reuse=None):
    #获取输入的depth
    input_dim=x.get_shape().as_list()[-1]
    with tf.name_scope(scope):
        with tf.name_scope('weight'):
            #构造weight
            weight=weight_variable([input_dim,output_dim])
            variable_summaries(weight)
        with tf.name_scope('bias'):
            bias=bias_variable([output_dim])
            variable_summaries(bias)
        with tf.name_scope('linear'):
            #计算wx+b
            preact=tf.matmul(x,weight)+bias
            #
            tf.summary.histogram('pre_activation',preact)
        #激活函数
        output=act(preact)
        #添加激活后的输出分布情况到summary
        tf.summary.histogram('output',output)
        return output
    
#构建DNN
def DNN(x,output_depths,scope='DNN_with_summary',reuse=None):
    with tf.name_scope(scope):
        net=x
        for i,output_depth in enumerate(output_depths):
            net = hidden_layer(net, output_depth, scope='hidden%d'% (i+1),reuse=reuse)
            #最后一个分类层
        net=hidden_layer(net,10,scope='classification',act=tf.identity,reuse=reuse)
        return net
dnn_with_sums=DNN(input_ph,[400,200,100])
#重新定义loss acc train_op
with tf.name_scope('cross_entropy'):
    loss=tf.losses.softmax_cross_entropy(logits=dnn_with_sums,onehot_labels=label_ph)
    tf.summary.scalar('cross_entropy',loss)
with tf.name_scope('accuracy'):
    acc=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(dnn_with_sums,axis=-1),tf.argmax(label_ph,axis=-1)),dtype=tf.float32))
    tf.summary.scalar('accuracy',acc)
with tf.name_scope('train'):
    lr=0.01
    optimizer=tf.train.GradientDescentOptimizer(learning_rate=lr)
    train_op=optimizer.minimize(loss)
    
#融合summary(可选)
#将之前所有的summary融合成一个summary
merged=tf.summary.merge_all()
sess=tf.InteractiveSession()
##输出summary 
##首先定义文件读写器
#summary_writer=tf.summary.FileWriter('summaries',sess.graph)
##？？？在希望的时候运行merged
#summaries = sess.run(merged, feed_dict={...})
##将summary写入summary_writer内
#summary_writer.add_summary(summaries,step)
##关闭文件读写器
#summary_writer.close()

train_writer=tf.summary.FileWriter('test_summary/train',sess.graph)
test_writer=tf.summary.FileWriter('test_summary/test',sess.graph)

#训练模型
batch_size=64
sess.run(tf.global_variables_initializer())
for e in range(20000):
    images,labels=train_set.next_batch(batch_size)
    sess.run(train_op,feed_dict={input_ph:images,label_ph:labels})
    if e%1000==999:
        test_imgs,test_labels=test_set.next_batch(batch_size)
        #获取train数据的summaries loss acc信息
        sum_train,loss_train,acc_train=sess.run([merged,loss,acc],feed_dict={input_ph:images,label_ph:labels})
        #将train的summaries写入train_writer
        train_writer.add_summary(sum_train,e)
        # 获取`test`数据的`summaries`以及`loss`, `acc`信息
        sum_test,loss_test,acc_test=sess.run([merged,loss,acc],feed_dict={input_ph:test_imgs,label_ph:test_labels})
        #将test的summaries写入train_writer
        test_writer.add_summary(sum_test,e)
        print('STEP{}: train_loss:{:.6f} train_acc:{:.6f} test_loss{:.6f} test_acc{:.6f}'.format(e+1,loss_train,acc_train,loss_test,acc_test))
train_writer.close()
test_writer.close()
print('Train done!')
print('-'*30)
    
              
    


        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        