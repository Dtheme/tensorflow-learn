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

#  get_variable() 在创建变量的时候会查看当前的命名空间里是否有这个name的变量 如果有直接用，如果没有会创建一个新的
#  variable()在创建的时候是重新创建，如果已经有了会报错（重定义）。
with tf.variable_scope("scope1"):
    w1 = tf.get_variable("w1", shape=[])
    w2 = tf.Variable(0.0, name="w2")
with tf.variable_scope("scope1", reuse=True):
    w1_p = tf.get_variable("w1", shape=[])
    w2_p = tf.Variable(1.0, name="w2")
    
print(w1 is w1_p, w2 is w2_p)

