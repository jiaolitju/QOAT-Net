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
#
# test_input_folder="E:\\定量光声\\Train_Save\\训练结果\\2\\0.01 test data\\归一化\\1.5\\p0.mat"

# data_test=load(test_input_folder)
# data_test = data_test['p0']
# data_test = np.reshape(data_test, [1,256,256,1])

#Parameters
INPUT_IMG_CHANNEL=1
learning_rate=1e-4 #1e-2开始还行，后面溢出？ 1e-3从5降到0.5
batch_size=4
training_epochs=200

X=tf.placeholder(tf.float32,[None,256,256,1])
Y_fai=tf.placeholder(tf.float32,[None,256,256,1])
Y_miua=tf.placeholder(tf.float32,[None,256,256,1])
Z=tf.placeholder(tf.float32,[None,256,256,1])
#读取tfrecords数据label,
def get_Batch(input,  batchsize, epochs):
	# 读取TFRecord文件，创建文件列表，并通过文件列表创建输入文件队列。
	# 在调用输入数据处理流程前，需要统一所有原始数据的格式并将它们存储到TFRecord文件中。
	#tf.train.match_filenames_once()：获取符合正则表达式的文件列表
	input_data = tf.train.match_filenames_once(input)
	# label_data = tf.train.match_filenames_once(label)

	input_queue = tf.train.string_input_producer(input_data, num_epochs=epochs, shuffle=False)  # 不随机打乱
	# label_queue = tf.train.string_input_producer(label_data, num_epochs=epochs, shuffle=False)

	# 创建一个reader来读取TFRecord文件中的样例。
	reader = tf.TFRecordReader()
	# reader从文件名队列中读一个样例。对应的方法是reader.read
	_, serialized_example = reader.read(input_queue) #返回文件名和文件

	#解析单个example
	features = tf.parse_single_example(serialized_example,
									   features={
										   'input_raw': tf.FixedLenFeature([], tf.string),
										   'true_raw1': tf.FixedLenFeature([], tf.string),
										   'true_raw2': tf.FixedLenFeature([], tf.string),
										   'true_raw3': tf.FixedLenFeature([], tf.string),
									   })
	# 得到原始数据、尺寸。
	input_data1=features['input_raw'] #
	label1_data1 = features['true_raw1']
	label2_data1 = features['true_raw2']
	label3_data1 = features['true_raw3']

	# _, serialized_example = reader.read(label_queue)
	# features = tf.parse_single_example(serialized_example,
	# 								   features={
	# 									   'true_raw': tf.FixedLenFeature([], tf.string),
	# 								   })

	#tf.decode_raw可以将字符串解析成图像对应的像素数组
	decode_input = tf.decode_raw(input_data1, tf.float32)
	decode_input = tf.reshape(decode_input, [256, 256, 1])
	# decode_input=tf.transpose(decode_input)
	# decode_input=tf.reshape(decode_input, [128, 128,1])

	decode_label1 = tf.decode_raw(label1_data1, tf.float32)  #当用miua时是float64,  fai是float32
	decode_label1 = tf.reshape(decode_label1, [256, 256, 1])
	# decode_label=tf.transpose(decode_label)
	# decode_label=tf.reshape(decode_label, [128, 128,1])

	decode_label2 = tf.decode_raw(label2_data1, tf.float32)  #当用miua时是float64,  fai是float32
	decode_label2 = tf.reshape(decode_label2, [256, 256, 1])

	decode_label3 = tf.decode_raw(label3_data1, tf.float32)  #当用miua时是float64,  fai是float32
	decode_label3 = tf.reshape(decode_label3, [256, 256, 1])


	# 进行批处理
	x_batch, y_batch1, y_batch2, y_batch3 = tf.train.shuffle_batch(([decode_input,decode_label1, decode_label2, decode_label3]), batch_size=batchsize,
											  num_threads=1, capacity=300,
											  min_after_dequeue=200) #
	return x_batch, y_batch1, y_batch2, y_batch3

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
def Unet_fai(x,batch_size):
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
		reglu4_1 =leaky_relu(tf.nn.bias_add(conv4_1, conv4_1_biases))

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

def Unet_miua(x,batch_size):
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


# y_pred=Unet(X,batch_size)  #获得的是神经网络的输出regul10

y_fai_pred=Unet_fai(X,batch_size)
y_miua_pred=Unet_miua(X,batch_size)

y_true=Z
y_fai_true=Y_fai
y_miua_true=Y_miua
y_pred_p0=tf.multiply(y_fai_pred, y_miua_pred)

#加高阶导数限制条件
#一阶导数
d1x,d1y=tf.image.image_gradients(y_fai_pred)  #Tensor("Reshape_1:0", shape=(1, 128, 128, 1), dtype=float32)
num1=tf.add(tf.pow(d1x,2), tf.pow(d1y,2))
d1=tf.sqrt(num1+1e-8)
u1x,u1y=tf.image.image_gradients(y_miua_pred)  #Tensor("Reshape_1:0", shape=(1, 128, 128, 1), dtype=float32)
unum1=tf.add(tf.pow(u1x,2), tf.pow(u1y,2))
u1=tf.sqrt(unum1+1e-8)
#
# tt1x,tt1y=tf.image.image_gradients(y_miua_true)  #Tensor("Reshape_1:0", shape=(1, 128, 128, 1), dtype=float32)
# ttnum1=tf.add(tf.pow(tt1x,2), tf.pow(tt1y,2))
# tt1=tf.sqrt(ttnum1+1e-8)
#二阶导
# d2x,d2y=tf.image.image_gradients(d1)  #Tensor("Reshape_1:0", shape=(1, 128, 128, 1), dtype=float32)
# num2=tf.add(tf.pow(d2x,2), tf.pow(d2y,2))
# d2=tf.sqrt(num2+1e-8)
# maxnum2=tf.reduce_max(d2)
#三阶导
# d3x,d3y=tf.image.image_gradients(d2)  #Tensor("Reshape_1:0", shape=(1, 128, 128, 1), dtype=float32)
# num3=tf.add(tf.pow(d3x,2), tf.pow(d3y,2))
# d3=tf.sqrt(num3+1e-8)
# maxnum3=tf.reduce_max(d3)

#真实的fai求一阶导数
t1x,t1y=tf.image.image_gradients(y_fai_true)  #Tensor("Reshape_1:0", shape=(1, 128, 128, 1), dtype=float32)
tnum1=tf.add(tf.pow(t1x,2), tf.pow(t1y,2))
t1=tf.sqrt(tnum1+1e-8)
ut1x,ut1y=tf.image.image_gradients(y_fai_true)  #Tensor("Reshape_1:0", shape=(1, 128, 128, 1), dtype=float32)
utnum1=tf.add(tf.pow(ut1x,2), tf.pow(ut1y,2))
ut1=tf.sqrt(utnum1+1e-8)
# t2x,t2y=tf.image.image_gradients(t1)  #Tensor("Reshape_1:0", shape=(1, 128, 128, 1), dtype=float32)
# tnum2=tf.add(tf.pow(t2x,2), tf.pow(t2y,2))
# t2=tf.sqrt(tnum2+1e-8)
# t3x,t3y=tf.image.image_gradients(t2)  #Tensor("Reshape_1:0", shape=(1, 128, 128, 1), dtype=float32)
# tnum3=tf.add(tf.pow(t3x,2), tf.pow(t3y,2))
# t3=tf.sqrt(tnum2+1e-8)

#错误率计算,用矩阵的F范数计算
fz=tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(y_fai_true-y_fai_pred))))
fm=tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(y_fai_true))))
fe=tf.divide(fz, fm)
fa=1-fe

uz=tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(y_miua_true-y_miua_pred))))
um=tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(y_miua_true))))
ue=tf.divide(uz, um)
ua=1-ue

#Define loss and optimizer, minimize the squared error
cost=tf.reduce_mean(tf.pow(y_true-y_pred_p0,2))+100*tf.reduce_mean(tf.pow(y_fai_true-y_fai_pred,2))\
	 +200*tf.reduce_mean(tf.pow(y_miua_true-y_miua_pred,2))
#+ (1e-5) * tf.sqrt(tf.reduce_sum(tf.square(d1-t1)))+ (1e-7) * tf.sqrt(tf.reduce_sum(tf.square(u1-ut1)))\
#+(1e-4)*tf.reduce_sum(tf.image.total_variation(y_miua_pred)) +(1e-7)*tf.reduce_sum(tf.image.total_variation(y_fai_pred))


optimizer=tf.train.AdamOptimizer(learning_rate).minimize(cost) #
# x_batch, y_batch = get_Batch(train_input_data, train_true_data, batch_size,training_epochs)
x_batch, y_batch1, y_batch2, y_batch3 = get_Batch("sys_train_data.tfrecords", batch_size, training_epochs)
# v_batch, v_batch1, v_batch2 = get_Batch("sys_valid_data.tfrecords", batch_size, training_epochs*13)
# print(x_batch.shape)

#保存模型 初始化TensorFlow持久化类
saver = tf.train.Saver()

#record cost
cost_all=[]
# cost_val=[]
facc_all=[]
# facc_val=[]
uacc_all=[]
# uacc_val=[]
#Launch the graph
with tf.Session() as sess:
	# Initilizaing the variables
	sess.run(tf.global_variables_initializer())
	sess.run(tf.local_variables_initializer())
	# saver.restore(sess, "H:\\sys cyclegan\\save\\2020-06-10\\1209一部分假实验加1119真实验4000张\\model.ckpt")
	#开启协调器
	coord = tf.train.Coordinator()
	# 使用start_queue_runners 启动内存队列填充
	threads = tf.train.start_queue_runners(sess=sess, coord=coord)
	ites=0
	try:
		while not coord.should_stop():
			data,label1,label2,label3=sess.run([x_batch,y_batch1, y_batch2, y_batch3]) #在会话中取出image和label
			# data_v, label1_v, label2_v = sess.run([v_batch, v_batch1, v_batch2])
			# c_v, facc_v, uacc_v = sess.run([cost, fa, ua], feed_dict={X: data_v, Y_fai: label1_v, Y_miua: label2_v})
			_, c, facc, uacc=sess.run([optimizer, cost, fa, ua],feed_dict={X:data, Y_fai:label1, Y_miua:label2,Z:label3})
			if ites%90==0:
				print("After %d training epoch(s), train cost is %g, uacc is %g" % (ites, c, uacc))

				cost_all.append(c)
				# cost_val.append(c_v)
				facc_all.append(facc)
				uacc_all.append(uacc)
				# facc_val.append(facc_v)
				# uacc_val.append(uacc_v)
			if ites%900==0:
				saver.save(sess, "E:\\210401程序材料\\data\\pork\\model\\model.ckpt", global_step=ites)
			ites=ites+1
	except tf.errors.OutOfRangeError:
		print("---Train end---")
	finally:
		# 协调器coord发出所有线程终止信号
		coord.request_stop()
	# 把开启的线程加入主线程，等待threads结束
	coord.join(threads)
	#####保存模型
	save_path = saver.save(sess, "E:\\210401程序材料\\data\\pork\\model\\model.ckpt")  # 保存会话中的所有内容？只保存参数模型

	# y_test=sess.run(y_miua_pred,feed_dict={X:data_test})
	# print(np.reshape(y_test,(128,128)))

	# plt.imshow(np.reshape(y_test, (128, 128)))
	# # #plt.imshow(np.reshape(y_test,(128,128)))
	# plt.colorbar()
	# plt.show()

	savemat("E:\\210401程序材料\\data\\pork\\model\\train_cost", {'tcost': cost_all})
	# savemat("E:\\定量光声\\Train_Save\\valid_cost", {'vcost': cost_val})
	savemat("E:\\210401程序材料\\data\\pork\\model\\tfacc", {'tfacc': facc_all})
	savemat("E:\\210401程序材料\\data\\pork\\model\\tuacc", {'tuacc': uacc_all})
	# savemat("E:\\定量光声\\Train_Save\\vfacc", {'vfacc': facc_val})
	# savemat("E:\\定量光声\\Train_Save\\vuacc", {'vuacc': uacc_val})
	# y_test=np.reshape(y_test, (128, 128))
	# savemat("E:\\定量光声\\Train_Save\\utest", {'utest': y_test})
	#
	#
	# y_test2=sess.run(y_fai_pred,feed_dict={X:data_test})
	# y_test2=np.reshape(y_test2, (128, 128))
	# savemat("E:\\定量光声\\Train_Save\\ftest", {'ftest': y_test2})
