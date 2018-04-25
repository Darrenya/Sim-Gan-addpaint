"""
Implementation of `3.1 Appearance-based Gaze Estimation` from
[Learning from Simulated and Unsupervised Images through Adversarial Training](https://arxiv.org/pdf/1612.07828v1.pdf).

Note: Only Python 3 support currently.
"""

import os
import sys

# from keras import applications
# from keras import layers
# from keras import models
# from keras import optimizers
# from keras.preprocessing import image
# import numpy as np
import tensorflow as tf
# import scipy.io as scio 
# from dlutils import plot_image_batch_w_labels##看github   加载的必须文件   作用百度

# from utils.image_history_buffer import ImageHistoryBuffer

#数据生成的代码
def sample_data(size, length=100):
	"""    随机mean=4 std=1.5的数据    :param size:    :param length:    :return:    """
	data = []
	for _ in range(size):
		data.append(sorted(np.random.normal(4, 1.5, length)))
	return np.array(data)
	#生成噪音的代码
	def random_data(size, length=100):
		"""    随机生成数据    :param size:    :param length:    :return:	"""
	data = []
	for _ in range(size):
		x = np.random.random(length) 
		data.append(x)
	return np.array(data)
	#在训练前需要对数据进行预处理
	def preprocess_data(x):
		"""    计算每一组数据平均值和方差    :param x:    :return:    """
	return [[np.mean(data), np.std(data)] for data in x]
	
	#G和D的连接之间也需要做出处理
	# 先求出G_output3的各行平均值和方差 
	# def G_output3(size, length=100):
		# """    随机mean=4 std=1.5的数据    :param size:    :param length:    :return:    """;
		# G_output3=[]
tf.expand_dims(tf.reduce_mean(G_output3, 1),1) # 平均值，但是是1D向量
MEAN_T = tf.transpose(tf.expand_dims(MEAN, 0))  # 转置
STD = tf.sqrt(tf.reduce_mean(tf.square(G_output3 - MEAN_T), 1))
DATA = tf.concat(1, [MEAN_T,
				tf.transpose(tf.expand_dims(STD, 0))]) # 拼接起来