import tensorflow as tf


# tf.placeholder是定义a，b的占位值，真实的值是后面feed_dict={}代入的
# c1 = a1 + b1定义了c1的表达式，但这个运算并为被激活，由session.run()激活
# 注意shape参数，它是tensor张量的一个属性用来表述数据的维度信息。
a1 = tf.placeholder(dtype=tf.float32, shape=None)
b1 = tf.placeholder(dtype=tf.float32, shape=None)
c1 = a1 + b1 

a2 = tf.placeholder(dtype=tf.float32, shape=None)
b2 = tf.placeholder(dtype=tf.float32, shape=None)
c2 = tf.matmul(a2, b2)


# 通过session激活运算并且将运算式子中的变量值传入
with tf.Session() as tSession:
    v_c1 = tSession.run(c1, feed_dict={
        a1:154,
        b1:28
    })
    v_c2 = tSession.run(c1, feed_dict={
        a1:154,
        b1:28
    })
print(v_c1, v_c2)
