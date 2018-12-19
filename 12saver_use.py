import tensorflow as tf
import numpy as np

# 提取保存的模型使用
# 需要定义跟上面同样shape和type的变量接受然后使用 创建几个placeholder值
W = tf.Variable([[0,0,0],[0,0,0]],dtype=tf.float32,name='wieghts') 
b = tf.Variable([0,0,0],dtype=tf.float32,name='biases')
# W = tf.Variable(np.arange(0, 6, 1).reshape((2,3)), dtype=tf.float32,name='wieghts') 
# b = tf.Variable(np.arange(0, 3, 1).reshape((1,3)),dtype=tf.float32,name='biases')
w_use = tf.Variable(np.arange(0, 6, 1).reshape())
print("-------:", W,b)
saver = tf.train.Saver()
with tf.Session() as tSession:
    saver.restore(tSession, "NetSaver/saver_net1.ckpt")
    print("Weights:", tSession.run(W))
    print("Biases:", tSession.run(b))
