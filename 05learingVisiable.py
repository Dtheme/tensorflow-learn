import tensorflow as tf
import numpy as np


def add_layer(inputs, in_size, out_size, n_layer, activation_function=None):   
    LayerName = 'layer%s' % n_layer
    with tf.name_scope(LayerName):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
            tf.summary.histogram(LayerName+'/Weights',Weights)
        with tf.name_scope("biases"):
            biases = tf.Variable(tf.zeros([1, out_size])+0.1, name='B')
            tf.summary.histogram(LayerName+'/biases',Weights)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)

        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
            tf.summary.histogram(LayerName+'/outputs',outputs)
        return outputs


x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.5, x_data.shape)
y_data = np.square(x_data)-0.5

with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 1],name='x_inputs')
    ys = tf.placeholder(tf.float32, [None, 1],name='y_inputs')


# hidden layer
layer1Result = add_layer(xs, 1, 10, n_layer=1, activation_function=tf.nn.relu)

# output layer
tPrediction = add_layer(layer1Result, 10, 1, n_layer=2, activation_function=None)

# loss函数
# tf.reduce_sum(tf.square(ys - tPrediction, reduction_indices=[1]))
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - tPrediction,name='square'),reduction_indices=[1], name='reduce_sum'), name='loss')
    tf.summary.scalar('loss',loss)
# 通过优化器设置learning rate / step 
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# init = tf.global_variables_initializer
tSession = tf.Session()
merged = tf.summary.merge_all()

writer = tf.summary.FileWriter("log/",tSession.graph)
tSession.run(tf.global_variables_initializer())

# 训练
for i in range(1000):
    tSession.run(train_step, feed_dict={
        xs: x_data,
        ys: y_data
    })
    if i % 50 == 0:
        print("loss值:%s" % (tSession.run(loss, feed_dict={xs: x_data, ys: y_data})))
        # print("预期值:%s" % (tSession.run(tPrediction,feed_dict={xs:x_data})))
        result = tSession.run(merged,feed_dict={
            xs:x_data,
            ys:y_data
        })
        writer.add_summary(result, i)
