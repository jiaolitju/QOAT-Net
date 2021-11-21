'''论文：Reconstruction of initial pressure from limited view photoacoustic images using deep learning
	前向传播模型,use tf.train.string_input_producer and tf.wholefilereader() to load pictures
	 这种方法加速了'''

# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from scipy.io import savemat
from scipy.io import loadmat as load
import time
#
test_input_folder="E:\\210401程序材料\\QOAT-Net\\Pork tissues\\input p0.mat"
data_test=load(test_input_folder)
data_test = data_test['p0']
data_test = np.reshape(data_test, [1,256,256,1])

#Parameters
INPUT_IMG_CHANNEL=1

X=tf.placeholder(tf.float32,[None,256,256,1])
Y_fai=tf.placeholder(tf.float32,[None,256,256,1])
Y_miua=tf.placeholder(tf.float32,[None,256,256,1])

def copy_and_crop_and_merge(contract_layer, upsampling):
	contract_layer_shape = tf.shape(contract_layer)
	upsampling_shape = tf.shape(upsampling)
	# tf.shape(a)获得Tensor a的尺寸，a可以是Tensor、list、array
	contract_layer_crop = contract_layer
	return tf.concat(values=[contract_layer_crop, upsampling], axis=-1) #axis=-1表示倒数第一个数。负数表示倒数#tf.concat,需要run
	#axis则是我们想要连接的维度,在行，列，通道上进行堆砌。0是行，1是列，2是通道
	# tf.concat返回的是连接后的tensor

def leaky_relu(a, alpha=0.1):
	a = tf.maximum(alpha * a, a)#tf.maximum(a,b),返回的是a,b之间的最大值
	return a

##定义卷积神经网络的前向传播过程。
def Unet_fai(x):
	# U-net主体结构
	# 论文中输入图像大小为1×128×128
	# layer 1
	##声明第一层卷积层的变量并实现前向传播过程。通过使用不同的命名空间来隔离不同层的变量，
	# 这可以让每一层中的变量命名只需要考虑在当前层的作用，而不需要担心重名的问题。
	with tf.variable_scope('layer1-conv1'):  # 通过tf.variable_scope函数可以控制tf.get_variable函数的语义.
		# conv1-1
		conv1_weights = tf.get_variable('weights', [3, 3, INPUT_IMG_CHANNEL, 16],
										initializer=tf.truncated_normal_initializer(stddev=0.1))
		# W的前两个数代表滤波器的尺寸，第三个数表示当前层的深度，第四个数表示过滤器的深度
		conv1_biases = tf.get_variable('bias', [16], initializer=tf.constant_initializer(0.0))
		# 使用边长为3，深度为32的滤波器，步长1×1，使用全零填充
		conv1 = tf.nn.conv2d(x, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')  # 第一和第四个参数必须为1
		reglu1 = leaky_relu(tf.nn.bias_add(conv1, conv1_biases))

		# conv1-2
		conv1_1_weights = tf.get_variable('weights_1', [3, 3, 16, 16],
										  initializer=tf.truncated_normal_initializer(stddev=0.1))
		# W的前两个数代表滤波器的尺寸，第三个数表示当前层的深度，第四个数表示过滤器的深度
		conv1_1_biases = tf.get_variable('bias_1', [16], initializer=tf.constant_initializer(0.0))
		# 使用边长为3，深度为32的滤波器，步长1×1，使用全零填充
		conv1_1 = tf.nn.conv2d(reglu1, conv1_1_weights, strides=[1, 1, 1, 1], padding='SAME')  # 第一和第四个参数必须为1
		reglu1_1 = leaky_relu(tf.nn.bias_add(conv1_1, conv1_1_biases))

		print(reglu1_1.get_shape())
		contract_layer1 = reglu1_1  # 该层结果临时保存, 供上采样使用（跳跃连接）
	# maxpool
	# 实现池化层的前向传播过程，最大池化，边长为2，全零填充，步长2
	with tf.name_scope('layer1-pool1'):
		maxpool1 = tf.nn.max_pool(reglu1_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # 最开始步长为2*2
		print(maxpool1.get_shape())

	# layer 2
	with tf.variable_scope('layer2-conv2'):
		# conv2-1
		conv2_weights = tf.get_variable('weights', [3, 3, 16, 32],
										initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv2_biases = tf.get_variable('bias', [32], initializer=tf.constant_initializer(0.0))
		conv2 = tf.nn.conv2d(maxpool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
		reglu2 = leaky_relu(tf.nn.bias_add(conv2, conv2_biases))

		# conv2-2
		conv2_1_weights = tf.get_variable('weights_1', [3, 3, 32, 32],
										  initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv2_1_biases = tf.get_variable('bias_1', [32], initializer=tf.constant_initializer(0.0))
		conv2_1 = tf.nn.conv2d(reglu2, conv2_1_weights, strides=[1, 1, 1, 1], padding='SAME')
		reglu2_1 = leaky_relu(tf.nn.bias_add(conv2_1, conv2_1_biases))

		print(reglu2_1.get_shape())
		contract_layer2 = reglu2_1  # 该层结果临时保存, 供上采样使用（跳跃连接）
	# maxpool
	# 实现池化层的前向传播过程，最大池化，边长为2，全零填充，步长2
	with tf.name_scope('layer2-pool2'):
		maxpool2 = tf.nn.max_pool(reglu2_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
		print(maxpool2.get_shape())

	# layer 3
	with tf.variable_scope('layer3-conv3'):
		# conv3-1
		conv3_weights = tf.get_variable('weights', [3, 3, 32, 64],
										initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv3_biases = tf.get_variable('bias', [64], initializer=tf.constant_initializer(0.0))
		conv3 = tf.nn.conv2d(maxpool2, conv3_weights, strides=[1, 1, 1, 1], padding='SAME')
		reglu3 = leaky_relu(tf.nn.bias_add(conv3, conv3_biases))

		# conv3-2
		conv3_1_weights = tf.get_variable('weights_1', [3, 3, 64, 64],
										  initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv3_1_biases = tf.get_variable('bias_1', [64], initializer=tf.constant_initializer(0.0))
		conv3_1 = tf.nn.conv2d(reglu3, conv3_1_weights, strides=[1, 1, 1, 1], padding='SAME')
		reglu3_1 = leaky_relu(tf.nn.bias_add(conv3_1, conv3_1_biases))

		print(reglu3_1.get_shape())
		contract_layer3 = reglu3_1  # 该层结果临时保存, 供上采样使用（跳跃连接）
	# maxpool
	# 实现池化层的前向传播过程，最大池化，边长为2，全零填充，步长2
	with tf.name_scope('layer3-pool3'):
		maxpool3 = tf.nn.max_pool(reglu3_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
		print(maxpool3.get_shape())

	# layer 4
	with tf.variable_scope('layer4-conv4'):
		# conv4-1
		conv4_weights = tf.get_variable('weights', [3, 3, 64, 128],
										initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv4_biases = tf.get_variable('bias', [128], initializer=tf.constant_initializer(0.0))
		conv4 = tf.nn.conv2d(maxpool3, conv4_weights, strides=[1, 1, 1, 1], padding='SAME')
		reglu4 = leaky_relu(tf.nn.bias_add(conv4, conv4_biases))

		# conv4-2
		conv4_1_weights = tf.get_variable('weights_1', [3, 3, 128, 128],
										  initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv4_1_biases = tf.get_variable('bias_1', [128], initializer=tf.constant_initializer(0.0))
		conv4_1 = tf.nn.conv2d(reglu4, conv4_1_weights, strides=[1, 1, 1, 1], padding='SAME')
		reglu4_1 = leaky_relu(tf.nn.bias_add(conv4_1, conv4_1_biases))

		print(reglu4_1.get_shape())
		contract_layer4 = reglu4_1  # 该层结果临时保存, 供上采样使用（跳跃连接）
	# maxpool
	# 实现池化层的前向传播过程，最大池化，边长为2，全零填充，步长2
	with tf.name_scope('layer4-pool4'):
		maxpool4 = tf.nn.max_pool(reglu4_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
		print(maxpool4.get_shape())

	# layer 5 (bottom)
	with tf.variable_scope('layer5_conv5'):
		# conv5-1
		conv5_weights = tf.get_variable('weights', [3, 3, 128, 256],
										initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv5_biases = tf.get_variable('bias', [256], initializer=tf.constant_initializer(0.0))
		conv5 = tf.nn.conv2d(maxpool4, conv5_weights, strides=[1, 1, 1, 1], padding='SAME')
		reglu5 = leaky_relu(tf.nn.bias_add(conv5, conv5_biases))

		# conv5-2
		conv5_1_weights = tf.get_variable('weights_1', [3, 3, 256, 256],
										  initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv5_1_biases = tf.get_variable('bias_1', [256], initializer=tf.constant_initializer(0.0))
		conv5_1 = tf.nn.conv2d(reglu5, conv5_1_weights, strides=[1, 1, 1, 1], padding='SAME')
		reglu5_1 = leaky_relu(tf.nn.bias_add(conv5_1, conv5_1_biases))
		print(reglu5_1.get_shape())

	# up sample
	with tf.name_scope('layer5-up1'):
		# conv5_2_weights = tf.get_variable('weights_up1', [2, 2, 256, 128],
		# 								initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv5_2_biases = tf.get_variable('bias_up1', [128], initializer=tf.constant_initializer(0.0))
		# up_sample1 = tf.nn.conv2d_transpose(reglu4, conv5_weights, output_shape=[batch_size, 32, 32, 128],
		# 									strides=[1, 2, 2, 1], padding='SAME')  ##output_shape维度？？ filter的维度相反
		up_sample1 = tf.layers.conv2d_transpose(reglu5_1, 128, 2, strides=2,
												padding='SAME')  # 128是filters个数(通道数)，即output shape
		reglu5_2 = leaky_relu(tf.nn.bias_add(up_sample1, conv5_2_biases))
		print(reglu5_2.get_shape())

	# layer 6
	with tf.variable_scope('layer6-conv6'):
		# copy, crop and merge  （跳跃连接层）
		merge1 = copy_and_crop_and_merge(contract_layer=contract_layer4, upsampling=reglu5_2)
		# copy_and_crop_and_merge是自定义的函数（本程序还未进行定义）

		# conv6-1
		conv6_weights = tf.get_variable('weights', [3, 3, 256, 128],
										initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv6_biases = tf.get_variable('bias', [128], initializer=tf.constant_initializer(0.0))
		conv6 = tf.nn.conv2d(merge1, conv6_weights, strides=[1, 1, 1, 1], padding='SAME')
		reglu6 = leaky_relu(tf.nn.bias_add(conv6, conv6_biases))

		# conv6-2
		conv6_1_weights = tf.get_variable('weights_1', [3, 3, 128, 64],
										  initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv6_1_biases = tf.get_variable('bias_1', [64], initializer=tf.constant_initializer(0.0))
		conv6_1 = tf.nn.conv2d(reglu6, conv6_1_weights, strides=[1, 1, 1, 1], padding='SAME')
		reglu6_1 = leaky_relu(tf.nn.bias_add(conv6_1, conv6_1_biases))
		print(reglu6_1.get_shape())

	# up sample
	with tf.name_scope('layer6-up2'):
		# conv7_weights = tf.get_variable('weights_up2', [2, 2, 64, 64],
		# 								initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv6_2_biases = tf.get_variable('bias_up2', [64], initializer=tf.constant_initializer(0.0))
		# up_sample2 = tf.nn.conv2d_transpose(reglu6, conv7_weights, output_shape=[batch_size, 64, 64, 64],
		# 									strides=[1, 2, 2, 1], padding='SAME')  ##output_shape
		up_sample2 = tf.layers.conv2d_transpose(reglu6_1, 64, 2, strides=2, padding='SAME')
		reglu6_2 = leaky_relu(tf.nn.bias_add(up_sample2, conv6_2_biases))
	# print(reglu6_2.get_shape())

	# layer 7
	with tf.variable_scope('layer7_conv7'):
		# copy, crop and merge
		merge2 = copy_and_crop_and_merge(contract_layer=contract_layer3, upsampling=reglu6_2)
		# conv7-1
		conv7_weights = tf.get_variable('weights', [3, 3, 128, 64],
										initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv7_biases = tf.get_variable('bias', [64], initializer=tf.constant_initializer(0.0))
		conv7 = tf.nn.conv2d(merge2, conv7_weights, strides=[1, 1, 1, 1], padding='SAME')
		reglu7 = leaky_relu(tf.nn.bias_add(conv7, conv7_biases))

		# conv7-2
		conv7_1_weights = tf.get_variable('weights_1', [3, 3, 64, 32],
										  initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv7_1_biases = tf.get_variable('bias_1', [32], initializer=tf.constant_initializer(0.0))
		conv7_1 = tf.nn.conv2d(reglu7, conv7_1_weights, strides=[1, 1, 1, 1], padding='SAME')
		reglu7_1 = leaky_relu(tf.nn.bias_add(conv7_1, conv7_1_biases))
		print(reglu7_1.get_shape())

	# up sample
	with tf.name_scope('layer7-up3'):
		# conv7_2_weights = tf.get_variable('weights_up3', [2, 2, 32, 32],
		# 									initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv7_2_biases = tf.get_variable('bias_up3', [32], initializer=tf.constant_initializer(0.0))
		# up_sample3 = tf.nn.conv2d_transpose(reglu8, conv9_weights, output_shape=[batch_size, 128, 128, 32],
		# 										strides=[1, 2, 2, 1], padding='SAME')  ##output_shape
		up_sample3 = tf.layers.conv2d_transpose(reglu7_1, 32, 2, strides=2, padding='SAME')
		reglu7_2 = leaky_relu(tf.nn.bias_add(up_sample3, conv7_2_biases))
		print(reglu7_2.get_shape())

	# layer 8
	with tf.variable_scope('layer8_conv8'):
		# copy, crop and merge
		merge3 = copy_and_crop_and_merge(contract_layer=contract_layer2, upsampling=reglu7_2)
		# conv8-1
		conv8_weights = tf.get_variable('weights', [3, 3, 64, 32],
										initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv8_biases = tf.get_variable('bias', [32], initializer=tf.constant_initializer(0.0))
		conv8 = tf.nn.conv2d(merge3, conv8_weights, strides=[1, 1, 1, 1], padding='SAME')
		reglu8 = leaky_relu(tf.nn.bias_add(conv8, conv8_biases))

		# conv8-2
		conv8_1_weights = tf.get_variable('weights_1', [3, 3, 32, 16],
										  initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv8_1_biases = tf.get_variable('bias_1', [16], initializer=tf.constant_initializer(0.0))
		conv8_1 = tf.nn.conv2d(reglu8, conv8_1_weights, strides=[1, 1, 1, 1], padding='SAME')
		reglu8_1 = leaky_relu(tf.nn.bias_add(conv8_1, conv8_1_biases))
		print(reglu8_1.get_shape())

	# up sample
	with tf.name_scope('layer8-up4'):
		# conv8_2_weights = tf.get_variable('weights_up3', [2, 2, 32, 32],
		# 									initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv8_2_biases = tf.get_variable('bias_up4', [16], initializer=tf.constant_initializer(0.0))
		# up_sample3 = tf.nn.conv2d_transpose(reglu8, conv9_weights, output_shape=[batch_size, 128, 128, 32],
		# 										strides=[1, 2, 2, 1], padding='SAME')  ##output_shape
		up_sample4 = tf.layers.conv2d_transpose(reglu8_1, 16, 2, strides=2, padding='SAME')
		reglu8_2 = leaky_relu(tf.nn.bias_add(up_sample4, conv8_2_biases))
		print(reglu8_2.get_shape())

	# layer 9
	with tf.variable_scope('layer9_conv9'):
		# copy, crop and merge
		merge4 = copy_and_crop_and_merge(contract_layer=contract_layer1, upsampling=reglu8_2)
		# conv9-1
		conv9_weights = tf.get_variable('weights', [3, 3, 32, 16],
										initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv9_biases = tf.get_variable('bias', [16], initializer=tf.constant_initializer(0.0))
		conv9 = tf.nn.conv2d(merge4, conv9_weights, strides=[1, 1, 1, 1], padding='SAME')
		reglu9 = leaky_relu(tf.nn.bias_add(conv9, conv9_biases))

		# conv9-2
		conv9_1_weights = tf.get_variable('weights_1', [3, 3, 16, 1],
										  initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv9_1_biases = tf.get_variable('bias_1', [1], initializer=tf.constant_initializer(0.0))
		conv9_1 = tf.nn.conv2d(reglu9, conv9_1_weights, strides=[1, 1, 1, 1], padding='SAME')
		reglu9_1 = leaky_relu(tf.nn.bias_add(conv9_1, conv9_1_biases))
		print(reglu9_1.get_shape())

	return reglu9_1

def Unet_miua(x):
	############################另一条通路###############################################################

	# U-net主体结构
	# 论文中输入图像大小为1×128×128
	# layer 1
	##声明第一层卷积层的变量并实现前向传播过程。通过使用不同的命名空间来隔离不同层的变量，
	# 这可以让每一层中的变量命名只需要考虑在当前层的作用，而不需要担心重名的问题。
	with tf.variable_scope('layer21-conv1'):  # 通过tf.variable_scope函数可以控制tf.get_variable函数的语义.
		# conv1-1
		conv1_weights = tf.get_variable('weights', [3, 3, INPUT_IMG_CHANNEL, 16],
										initializer=tf.truncated_normal_initializer(stddev=0.1))
		# W的前两个数代表滤波器的尺寸，第三个数表示当前层的深度，第四个数表示过滤器的深度
		conv1_biases = tf.get_variable('bias', [16], initializer=tf.constant_initializer(0.0))
		# 使用边长为3，深度为32的滤波器，步长1×1，使用全零填充
		conv1 = tf.nn.conv2d(x, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')  # 第一和第四个参数必须为1
		reglu1 = leaky_relu(tf.nn.bias_add(conv1, conv1_biases))

		# conv1-2
		conv1_1_weights = tf.get_variable('weights_1', [3, 3, 16, 16],
										  initializer=tf.truncated_normal_initializer(stddev=0.1))
		# W的前两个数代表滤波器的尺寸，第三个数表示当前层的深度，第四个数表示过滤器的深度
		conv1_1_biases = tf.get_variable('bias_1', [16], initializer=tf.constant_initializer(0.0))
		# 使用边长为3，深度为32的滤波器，步长1×1，使用全零填充
		conv1_1 = tf.nn.conv2d(reglu1, conv1_1_weights, strides=[1, 1, 1, 1], padding='SAME')  # 第一和第四个参数必须为1
		reglu1_1 = leaky_relu(tf.nn.bias_add(conv1_1, conv1_1_biases))

		print(reglu1_1.get_shape())
		contract_layer1 = reglu1_1  # 该层结果临时保存, 供上采样使用（跳跃连接）
	# maxpool
	# 实现池化层的前向传播过程，最大池化，边长为2，全零填充，步长2
	with tf.name_scope('layer21-pool1'):
		maxpool1 = tf.nn.max_pool(reglu1_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # 最开始步长为2*2
		print(maxpool1.get_shape())

	# layer 2
	with tf.variable_scope('layer22-conv2'):
		# conv2-1
		conv2_weights = tf.get_variable('weights', [3, 3, 16, 32],
										initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv2_biases = tf.get_variable('bias', [32], initializer=tf.constant_initializer(0.0))
		conv2 = tf.nn.conv2d(maxpool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
		reglu2 = leaky_relu(tf.nn.bias_add(conv2, conv2_biases))

		# conv2-2
		conv2_1_weights = tf.get_variable('weights_1', [3, 3, 32, 32],
										  initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv2_1_biases = tf.get_variable('bias_1', [32], initializer=tf.constant_initializer(0.0))
		conv2_1 = tf.nn.conv2d(reglu2, conv2_1_weights, strides=[1, 1, 1, 1], padding='SAME')
		reglu2_1 = leaky_relu(tf.nn.bias_add(conv2_1, conv2_1_biases))

		print(reglu2_1.get_shape())
		contract_layer2 = reglu2_1  # 该层结果临时保存, 供上采样使用（跳跃连接）
	# maxpool
	# 实现池化层的前向传播过程，最大池化，边长为2，全零填充，步长2
	with tf.name_scope('layer22-pool2'):
		maxpool2 = tf.nn.max_pool(reglu2_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
		print(maxpool2.get_shape())

	# layer 3
	with tf.variable_scope('layer23-conv3'):
		# conv3-1
		conv3_weights = tf.get_variable('weights', [3, 3, 32, 64],
										initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv3_biases = tf.get_variable('bias', [64], initializer=tf.constant_initializer(0.0))
		conv3 = tf.nn.conv2d(maxpool2, conv3_weights, strides=[1, 1, 1, 1], padding='SAME')
		reglu3 = leaky_relu(tf.nn.bias_add(conv3, conv3_biases))

		# conv3-2
		conv3_1_weights = tf.get_variable('weights_1', [3, 3, 64, 64],
										  initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv3_1_biases = tf.get_variable('bias_1', [64], initializer=tf.constant_initializer(0.0))
		conv3_1 = tf.nn.conv2d(reglu3, conv3_1_weights, strides=[1, 1, 1, 1], padding='SAME')
		reglu3_1 = leaky_relu(tf.nn.bias_add(conv3_1, conv3_1_biases))

		print(reglu3_1.get_shape())
		contract_layer3 = reglu3_1  # 该层结果临时保存, 供上采样使用（跳跃连接）
	# maxpool
	# 实现池化层的前向传播过程，最大池化，边长为2，全零填充，步长2
	with tf.name_scope('layer23-pool3'):
		maxpool3 = tf.nn.max_pool(reglu3_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
		print(maxpool3.get_shape())

	# layer 4
	with tf.variable_scope('layer24-conv4'):
		# conv4-1
		conv4_weights = tf.get_variable('weights', [3, 3, 64, 128],
										initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv4_biases = tf.get_variable('bias', [128], initializer=tf.constant_initializer(0.0))
		conv4 = tf.nn.conv2d(maxpool3, conv4_weights, strides=[1, 1, 1, 1], padding='SAME')
		reglu4 = leaky_relu(tf.nn.bias_add(conv4, conv4_biases))

		# conv4-2
		conv4_1_weights = tf.get_variable('weights_1', [3, 3, 128, 128],
										  initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv4_1_biases = tf.get_variable('bias_1', [128], initializer=tf.constant_initializer(0.0))
		conv4_1 = tf.nn.conv2d(reglu4, conv4_1_weights, strides=[1, 1, 1, 1], padding='SAME')
		reglu4_1 = leaky_relu(tf.nn.bias_add(conv4_1, conv4_1_biases))

		print(reglu4_1.get_shape())
		contract_layer4 = reglu4_1  # 该层结果临时保存, 供上采样使用（跳跃连接）
	# maxpool
	# 实现池化层的前向传播过程，最大池化，边长为2，全零填充，步长2
	with tf.name_scope('layer24-pool4'):
		maxpool4 = tf.nn.max_pool(reglu4_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
		print(maxpool4.get_shape())

	# layer 5 (bottom)
	with tf.variable_scope('layer25_conv5'):
		# conv5-1
		conv5_weights = tf.get_variable('weights', [3, 3, 128, 256],
										initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv5_biases = tf.get_variable('bias', [256], initializer=tf.constant_initializer(0.0))
		conv5 = tf.nn.conv2d(maxpool4, conv5_weights, strides=[1, 1, 1, 1], padding='SAME')
		reglu5 = leaky_relu(tf.nn.bias_add(conv5, conv5_biases))

		# conv5-2
		conv5_1_weights = tf.get_variable('weights_1', [3, 3, 256, 256],
										  initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv5_1_biases = tf.get_variable('bias_1', [256], initializer=tf.constant_initializer(0.0))
		conv5_1 = tf.nn.conv2d(reglu5, conv5_1_weights, strides=[1, 1, 1, 1], padding='SAME')
		reglu5_1 = leaky_relu(tf.nn.bias_add(conv5_1, conv5_1_biases))
		print(reglu5_1.get_shape())

	# up sample
	with tf.name_scope('layer25-up1'):
		# conv5_2_weights = tf.get_variable('weights_up1', [2, 2, 256, 128],
		# 								initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv5_2_biases = tf.get_variable('bias_up11', [128], initializer=tf.constant_initializer(0.0))
		# up_sample1 = tf.nn.conv2d_transpose(reglu4, conv5_weights, output_shape=[batch_size, 32, 32, 128],
		# 									strides=[1, 2, 2, 1], padding='SAME')  ##output_shape维度？？ filter的维度相反
		up_sample1 = tf.layers.conv2d_transpose(reglu5_1, 128, 2, strides=2,
												padding='SAME')  # 128是filters个数(通道数)，即output shape
		reglu5_2 = leaky_relu(tf.nn.bias_add(up_sample1, conv5_2_biases))
		print(reglu5_2.get_shape())

	# layer 6
	with tf.variable_scope('layer26-conv6'):
		# copy, crop and merge  （跳跃连接层）
		merge1 = copy_and_crop_and_merge(contract_layer=contract_layer4, upsampling=reglu5_2)
		# copy_and_crop_and_merge是自定义的函数（本程序还未进行定义）

		# conv6-1
		conv6_weights = tf.get_variable('weights', [3, 3, 256, 128],
										initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv6_biases = tf.get_variable('bias', [128], initializer=tf.constant_initializer(0.0))
		conv6 = tf.nn.conv2d(merge1, conv6_weights, strides=[1, 1, 1, 1], padding='SAME')
		reglu6 = leaky_relu(tf.nn.bias_add(conv6, conv6_biases))

		# conv6-2
		conv6_1_weights = tf.get_variable('weights_1', [3, 3, 128, 64],
										  initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv6_1_biases = tf.get_variable('bias_1', [64], initializer=tf.constant_initializer(0.0))
		conv6_1 = tf.nn.conv2d(reglu6, conv6_1_weights, strides=[1, 1, 1, 1], padding='SAME')
		reglu6_1 = leaky_relu(tf.nn.bias_add(conv6_1, conv6_1_biases))
		print(reglu6_1.get_shape())

	# up sample
	with tf.name_scope('layer26-up2'):
		# conv7_weights = tf.get_variable('weights_up2', [2, 2, 64, 64],
		# 								initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv6_2_biases = tf.get_variable('bias_up21', [64], initializer=tf.constant_initializer(0.0))
		# up_sample2 = tf.nn.conv2d_transpose(reglu6, conv7_weights, output_shape=[batch_size, 64, 64, 64],
		# 									strides=[1, 2, 2, 1], padding='SAME')  ##output_shape
		up_sample2 = tf.layers.conv2d_transpose(reglu6_1, 64, 2, strides=2, padding='SAME')
		reglu6_2 = leaky_relu(tf.nn.bias_add(up_sample2, conv6_2_biases))
	# print(reglu6_2.get_shape())

	# layer 7
	with tf.variable_scope('layer27_conv7'):
		# copy, crop and merge
		merge2 = copy_and_crop_and_merge(contract_layer=contract_layer3, upsampling=reglu6_2)
		# conv7-1
		conv7_weights = tf.get_variable('weights', [3, 3, 128, 64],
										initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv7_biases = tf.get_variable('bias', [64], initializer=tf.constant_initializer(0.0))
		conv7 = tf.nn.conv2d(merge2, conv7_weights, strides=[1, 1, 1, 1], padding='SAME')
		reglu7 = leaky_relu(tf.nn.bias_add(conv7, conv7_biases))

		# conv7-2
		conv7_1_weights = tf.get_variable('weights_1', [3, 3, 64, 32],
										  initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv7_1_biases = tf.get_variable('bias_1', [32], initializer=tf.constant_initializer(0.0))
		conv7_1 = tf.nn.conv2d(reglu7, conv7_1_weights, strides=[1, 1, 1, 1], padding='SAME')
		reglu7_1 = leaky_relu(tf.nn.bias_add(conv7_1, conv7_1_biases))
		print(reglu7_1.get_shape())

	# up sample
	with tf.name_scope('layer27-up3'):
		# conv7_2_weights = tf.get_variable('weights_up3', [2, 2, 32, 32],
		# 									initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv7_2_biases = tf.get_variable('bias_up31', [32], initializer=tf.constant_initializer(0.0))
		# up_sample3 = tf.nn.conv2d_transpose(reglu8, conv9_weights, output_shape=[batch_size, 128, 128, 32],
		# 										strides=[1, 2, 2, 1], padding='SAME')  ##output_shape
		up_sample3 = tf.layers.conv2d_transpose(reglu7_1, 32, 2, strides=2, padding='SAME')
		reglu7_2 = leaky_relu(tf.nn.bias_add(up_sample3, conv7_2_biases))
		print(reglu7_2.get_shape())

	# layer 8
	with tf.variable_scope('layer28_conv8'):
		# copy, crop and merge
		merge3 = copy_and_crop_and_merge(contract_layer=contract_layer2, upsampling=reglu7_2)
		# conv8-1
		conv8_weights = tf.get_variable('weights', [3, 3, 64, 32],
										initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv8_biases = tf.get_variable('bias', [32], initializer=tf.constant_initializer(0.0))
		conv8 = tf.nn.conv2d(merge3, conv8_weights, strides=[1, 1, 1, 1], padding='SAME')
		reglu8 = leaky_relu(tf.nn.bias_add(conv8, conv8_biases))

		# conv8-2
		conv8_1_weights = tf.get_variable('weights_1', [3, 3, 32, 16],
										  initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv8_1_biases = tf.get_variable('bias_1', [16], initializer=tf.constant_initializer(0.0))
		conv8_1 = tf.nn.conv2d(reglu8, conv8_1_weights, strides=[1, 1, 1, 1], padding='SAME')
		reglu8_1 = leaky_relu(tf.nn.bias_add(conv8_1, conv8_1_biases))
		print(reglu8_1.get_shape())

	# up sample
	with tf.name_scope('layer28-up4'):
		# conv8_2_weights = tf.get_variable('weights_up3', [2, 2, 32, 32],
		# 									initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv8_2_biases = tf.get_variable('bias_up41', [16], initializer=tf.constant_initializer(0.0))
		# up_sample3 = tf.nn.conv2d_transpose(reglu8, conv9_weights, output_shape=[batch_size, 128, 128, 32],
		# 										strides=[1, 2, 2, 1], padding='SAME')  ##output_shape
		up_sample4 = tf.layers.conv2d_transpose(reglu8_1, 16, 2, strides=2, padding='SAME')
		reglu8_2 = leaky_relu(tf.nn.bias_add(up_sample4, conv8_2_biases))
		print(reglu8_2.get_shape())

	# layer 9
	with tf.variable_scope('layer29_conv9'):
		# copy, crop and merge
		merge4 = copy_and_crop_and_merge(contract_layer=contract_layer1, upsampling=reglu8_2)
		# conv9-1
		conv9_weights = tf.get_variable('weights', [3, 3, 32, 16],
										initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv9_biases = tf.get_variable('bias', [16], initializer=tf.constant_initializer(0.0))
		conv9 = tf.nn.conv2d(merge4, conv9_weights, strides=[1, 1, 1, 1], padding='SAME')
		reglu9 = leaky_relu(tf.nn.bias_add(conv9, conv9_biases))

		# conv9-2
		conv9_1_weights = tf.get_variable('weights_1', [3, 3, 16, 1],
										  initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv9_1_biases = tf.get_variable('bias_1', [1], initializer=tf.constant_initializer(0.0))
		conv9_1 = tf.nn.conv2d(reglu9, conv9_1_weights, strides=[1, 1, 1, 1], padding='SAME')
		reglu9_1 = leaky_relu(tf.nn.bias_add(conv9_1, conv9_1_biases))
		print(reglu9_1.get_shape())

	return reglu9_1


y_fai_pred=Unet_fai(X)
y_miua_pred=Unet_miua(X)

#保存模型 初始化TensorFlow持久化类
saver = tf.train.Saver()


#Launch the graph
with tf.Session() as sess:
	sess.run(tf.local_variables_initializer())
	saver.restore(sess, "E:\\210401程序材料\\data\\pork\\model\\model.ckpt")
	start = time.clock()
	y_test=sess.run(y_miua_pred,feed_dict={X:data_test})
	end = time.clock()
	t=end-start
	print("Runtime is:", t)

	y_test=np.reshape(y_test, (256, 256))

	plt.imshow(np.reshape(y_test, (256,256)))
	plt.colorbar()
	plt.show()

	savemat("E:\\210401程序材料\\save_test\\ua", {'ua': y_test})

	# y_test2=sess.run(y_fai_pred,feed_dict={X:data_test})
	# y_test2=np.reshape(y_test2, (256, 256))
	# savemat("E:\\210401程序材料\\save_test\\fai", {'fai': y_test2})

