import tensorflow as tf
import numpy as np

# 提取保存的模型使用
# 需要定义跟上面同样shape和type的变量接受然后使用 创建几个placeholder值
w_use = tf.Variable([[0,0,0],[0,0,0]],dtype=tf.float32,name='wieghts') 
b_use = tf.Variable([0,0,0],dtype=tf.float32,name='biases')

print(w_use, b_use)
print("======", np.arange(0, 6).reshape((2,3)), "-----", np.arange(0, 3).reshape((1,3)))
saver = tf.train.Saver()
with tf.Session() as tSession:
    saver.restore(tSession, "NetSaver/saver_net1.ckpt")
    print("Weights:", tSession.run(w_use))
    print("Biases:", tSession.run(b_use))
