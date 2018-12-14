
import tensorflow as tf

# tensorflow常量运算 
a = tf.constant = [1, 2] 
b = tf.constant = 100
vMul = tf.multiply(a, b)

# 运算得到tensor值
# 使用 with .. as .. 可以省点控制session的作用域 不需要主动close()
# vMul并没有激活运算，需要通过session.run()激活运算
with tf.Session() as tSession:
    TensorResult = tSession.run(vMul)
    print(vMul)

# 另一种使用session的方式
tSession2 = tf.Session()
tsResult = tSession2.run(vMul)
print(vMul)
tSession.close()
