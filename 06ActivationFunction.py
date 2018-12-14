import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 假数据 创建【-5，5】之间的200个数据的等差数列的数据
x = np.linspace(-10, 10, 400)

# 常用的activation function（激活函数），由tf.nn提供
# 数学基础：
# relu：r(x)=max(0,x)
# sigmod：φ(x)=1/(1+e^(-x))
# tanh：tanh(x)=(e^x - e^(-x))/(e^x + e^(-x))
# softplus：ζ(x)=log(1+e^x) 平滑版的relu

func_relu = tf.nn.relu(x)
func_sigmod = tf.nn.sigmoid(x)
func_tanh = tf.nn.tanh(x)
func_softplus = tf.nn.softplus(x)

with tf.Session() as tSession:
    func_relu = tSession.run(func_relu)
    func_sigmod = tSession.run(func_sigmod)
    func_tanh = tSession.run(func_tanh)
    func_softplus = tSession.run(func_softplus)

plt.figure(1, figsize=(8, 6))
plt.subplot(221)
plt.plot(x, func_relu, c='red', label='relu')
plt.ylim((-1, 5))
plt.legend(loc='best')

plt.subplot(222)
plt.plot(x, func_sigmod, c='red', label='sigmoid')
plt.ylim((-0.2, 1.2))
plt.legend(loc='best')

plt.subplot(223)
plt.plot(x, func_tanh, c='red', label='tanh')
plt.ylim((-1.2, 1.2))
plt.legend(loc='best')

plt.subplot(224)
plt.plot(x, func_softplus, c='red', label='softplus')
plt.ylim((-0.2, 6))
plt.legend(loc='best')

plt.show()
