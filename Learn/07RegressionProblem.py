"""
线性回归例子可视化
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tf.set_random_seed(1)
np.random.seed(1)

# 假数据 [-1,1]之间的等差数列的数组转换成一个100行1列的矩阵
x = np.linspace(-1, 1, 100)[:,np.newaxis]
noise = np.random.normal(0, 0.1, size=x.shape)
# 构建一个函数y = x^2+ noise
y = np.power(x, 2) + noise

# 使用上面的点绘制函数图像 
plt.scatter(x, y)
plt.show()

# 构建一个简单的训练
tf_data_x = tf.placeholder(tf.float32, x.shape)
tf_data_y = tf.placeholder(tf.float32, y.shape)

layer = tf.layers.dense(tf_data_x, 10, tf.nn.relu)
output = tf.layers.dense(layer, 1)                  
loss = tf.losses.mean_squared_error(tf_data_y, output)   # compute cost
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.6)
train_op = optimizer.minimize(loss)

tSession = tf.Session()
tSession.run(tf.global_variables_initializer())

plt.ion()

for step in range(100):
    # train and net output
    _, l, pred = tSession.run([train_op, loss, output], {tf_data_x: x, tf_data_y: y})
    if step % 5 == 0:
        # plot and show learning process
        plt.cla()
        plt.scatter(x, y)
        plt.plot(x, pred, 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % l, fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()
