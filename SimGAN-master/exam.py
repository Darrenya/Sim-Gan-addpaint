import tensorflow as tf #便与书写
import numpy as np #科学计算的模块
x_data = np.random.rand(100).astype(np.float32)
y_data=x_data*0.1+0.3

###开始创建结构###
Weights = tf.Variable(tf.random_uniform([1],-1.0,1.0))
biases=tf.Variable(tf.zeros([1]))

y = Weights*x_data + biases

loss=tf.reduce_mean(tf.square(y-y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()
###结束###

##初始化结构
sess = tf.Session()
sess.run(init)

for step in range(201):
    sess.run(train)
    if step % 20 == 0:
      print(step,sess.run(Weights),sess.run(biases))