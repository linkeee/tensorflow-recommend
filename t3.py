import tensorflow as tf
import numpy as np

# 模拟生成100对数据对，对应的函数为y=x*0.1+0.3
x_data = np.random.rand(100).astype('float32')
y_data = x_data * 0.1 + 0.3

# 指定W和b变量的取值范围（用tensorflow来得到W和b）
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b

# 最小化均方误差
loss = tf.reduce_mean(tf.square(y - y_data))
optimzer = tf.train.GradientDescentOptimizer(0.5)
train = optimzer.minimize(loss)

# 初始化tensorflow参数
init = tf.initialize_all_variables()

# 运行数据流图（计算过程在这一步才开始）
sess = tf.Session()
sess.run(init)

# 观察多次迭代计算时，W和b的拟合值
sum = 0
for step in range(201):
    l, w = sess.run([loss, W])
    print('loss is %s,%s' % (l, w))
    # sess.run(train)
    sum += loss
    # if step % 20 == 0:
    #     print(step, sess.run(W), sess.run(b), sess.run(loss))

print(sess.run(sum / 201))
