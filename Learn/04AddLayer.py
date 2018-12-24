import tensorflow as tf
import numpy as np


def add_layer(inputs, in_size, out_size, activation_function=None):   
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size])+0.1)
    Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


# 生成伪数据
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.5, x_data.shape)
y_data = np.square(x_data)-0.5

xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])


# 添加隐藏层和输出层
# hidden layer
layer1Result = add_layer(xs, 1, 10, activation_function=tf.nn.relu)

# output layer
tPrediction = add_layer(layer1Result, 10, 1, activation_function=None)

# loss函数 
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - tPrediction),reduction_indices=[1]))
tf.summary.scalar('loss',loss)

# 通过优化器设置learning rate / step 
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# 初始化session 激活前面注册的变量
tSession = tf.Session()
tSession.run(tf.global_variables_initializer())

# 训练
for i in range(1000):
    tSession.run(train_step, feed_dict={
        xs: x_data,
        ys: y_data
    })
    if i % 50 == 0:
        # 打印loss的值，注意需要run一下
        print("loss值:%s" % (tSession.run(loss, feed_dict={xs: x_data, ys: y_data})))
