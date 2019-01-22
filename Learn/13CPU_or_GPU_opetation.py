import tensorflow as tf

#  "/cpu:0"：机器的 CPU。
#  "/device:GPU:0"：机器的 GPU（如果有一个）。
#  "/device:GPU:1"：机器的第二个 GPU（以此类推）。

#  手动定义操作执行的cpu
with tf.device('/cpu:0'):
        a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
        b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
        c = tf.matmul(a, b)

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
print(sess.run(c))

#  指定GPU执行操作
with tf.device("/GPU:0"):
    v3 = tf.constant(3)

#  指定分配GPU 50%的显存做运算
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5

#  不提前分配好显存大小 而是需要多少就给多少 按照实际使用来
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    print(v3)
