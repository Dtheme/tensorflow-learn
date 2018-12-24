"""
使用saver保存数据
"""
import tensorflow as tf


W = tf.Variable([[1,2,3],[3,4,5]],dtype=tf.float32,name='wieghts') 
b = tf.Variable([1,2,3],dtype=tf.float32,name='biases')

init = tf.global_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as tSession:
    tSession.run(init)
    saver_path = saver.save(tSession,"NetSaver/saver_net1.ckpt")
    print("保存成功", saver_path)
