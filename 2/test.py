# -*- coding: utf-8 -*-
import tensorflow as tf
mat=tf.constant([[1,0],[0,1]],name='matrix')
sess=tf.Session()
#print(sess.run(mat))
#print('tensor:{}'.format(mat.name),sess.run(mat))
x=tf.constant(2.0,dtype=tf.float32)
sigmod=tf.divide(1.0,tf.add(1.0,tf.exp(-x)))
#print(sess.run(sigmod))
with tf.name_scope('name_scope'):
    var_a=tf.Variable(0,dtype=tf.int32)
    var_b=tf.Variable([1,2],dtype=tf.float32)
#print(var_a.name)
#print(var_b.name)
a=tf.placeholder(tf.float32)
b=tf.square(a)
#print(sess.run(b,feed_dict={a:[1,2,4,8]}))
#for i in [1,2,4,8]:
#    print(sess.run(b,feed_dict={a:i}))
    #print(b.eval(feed_dict={a:i}))
#graph
#g=tf.get_default_graph()
#print(g)
#for op in g.get_operations():
#    print(op.name)
#a=g.get_tensor_by_name('Hello:0')
#print(a.eval())
g1=tf.Graph()
#print(g1)
print('defalt_graph',tf.get_default_graph())
g1.as_default()
print('defalt_graph',tf.get_default_graph())
a1=tf.constant(32,name='a1')
with g1.as_default():
    a2=tf.constant(32,name='a2')
print('a1.graph',a1.graph)
print('a2.graph',a2.graph)
with tf.Session() as sess:
    graph_writer=tf.summary.FileWriter('.',sess.graph)