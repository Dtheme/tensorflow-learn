import tensorflow as tf

# 一些基本的类似数据
# 创建一个维度为１, 类型为int的对象
data1 = tf.zeros([1], dtype=tf.int32)
print(data1)
# 创建一个维度为3, 类型为int的对象
data2 = tf.zeros([1,2,1], dtype=tf.int32)
print(data2)
# double
data3 = tf.zeros([8], dtype=tf.double)
print(data3)
# float
data4 = tf.zeros([8], dtype=tf.float16)
print(data4)

# 定义一个变量tf_var 默认值为0
tf_var = tf.Variable(0)

# 定义一个累加运算
# tf.assign（x，value）的意思是 把x的值设置为value
vAdd = tf.add(tf_var, 1)
update_vAdd = tf.assign(tf_var, vAdd)


# 注意 前面定义的变量是通过sess.run(tf.global_variables_initializer())初始化激活的
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(3):
        sess.run(update_vAdd)
        print(sess.run(tf_var))

 
，