import tensorflow as tf
import numpy as np
#保存
#记得存储时定义相同的dtype和shape
# W = tf.Variable([[1,2,3],[3,4,5]],dtype=tf.float32,name='weights')
# b = tf.Variable([[1,2,3]],dtype=tf.float32,name='biases')
# init=tf.initialize_all_variables()

# saver = tf.train.Saver()

# with tf.Session() as sess:
	# sess.run(init)
	# save_path=saver.save(sess,"C:\log\save.ckpt")
	# print("Save to path:",save_path)
	##提取
	#重新存储 变量
	#为变量重新定义相同的shape和type
W = tf.Variable(np.arange(6).reshape((2, 3)), dtype=tf.float32, name="weights")
b = tf.Variable(np.arange(3).reshape((1, 3)), dtype=tf.float32, name="biases")
#不用定义init step
saver = tf.train.Saver()
with tf.Session() as sess:
	saver.restore(sess,"C:\log\save.ckpt")
	print("weights:",sess.run(W))
	print("biaes:",sess.run(b))