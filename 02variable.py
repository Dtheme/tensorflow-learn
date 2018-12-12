import tensorflow as tf

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
