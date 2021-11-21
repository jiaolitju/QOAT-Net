import numpy as np
import tensorflow as tf
import math


def make_var(name, shape, trainable=True):
	return tf.get_variable(name, shape, trainable=trainable)

def conv2d(input_, output_dim, kernel_size, stride, padding="SAME", name="conv2d", biased=False):
	input_dim = input_.get_shape()[-1]
	with tf.variable_scope(name):
		kernel = make_var(name='weights', shape=[kernel_size, kernel_size, input_dim, output_dim])
		output = tf.nn.conv2d(input_, kernel, [1, stride, stride, 1], padding=padding)
		if biased:
			biases = make_var(name='biases', shape=[output_dim])
			output = tf.nn.bias_add(output, biases)
		return output

def atrous_conv2d(input_, output_dim, kernel_size, dilation, padding="SAME", name="atrous_conv2d", biased=False):
	input_dim = input_.get_shape()[-1]
	with tf.variable_scope(name):
		kernel = make_var(name='weights', shape=[kernel_size, kernel_size, input_dim, output_dim])
		output = tf.nn.atrous_conv2d(input_, kernel, dilation, padding=padding)
		if biased:
			biases = make_var(name='biases', shape=[output_dim])
			output = tf.nn.bias_add(output, biases)
		return output

def deconv2d(input_, output_dim, kernel_size, stride, padding="SAME", name="deconv2d"):
	input_dim = input_.get_shape()[-1]
	input_height = int(input_.get_shape()[1])
	input_width = int(input_.get_shape()[2])
	with tf.variable_scope(name):
		kernel = make_var(name='weights', shape=[kernel_size, kernel_size, output_dim, input_dim])
		# output = tf.nn.conv2d_transpose(input_, kernel, [1, input_height * 2, input_width * 2, output_dim],
		# 								[1, 2, 2, 1], padding="SAME")
		output = tf.layers.conv2d_transpose(input_, output_dim, 2, strides=2, padding='SAME')
		return output

def batch_norm(input_, name="batch_norm"):
	with tf.variable_scope(name):
		input_dim = input_.get_shape()[-1]
		scale = tf.get_variable("scale", [input_dim],
								initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
		offset = tf.get_variable("offset", [input_dim], initializer=tf.constant_initializer(0.0))
		mean, variance = tf.nn.moments(input_, axes=[1, 2], keep_dims=True)
		epsilon = 1e-5
		inv = tf.rsqrt(variance + epsilon)
		normalized = (input_ - mean) * inv
		output = scale * normalized + offset
		return output

def max_pooling(input_, kernel_size, stride, name, padding="SAME"):
	return tf.nn.max_pool(input_, ksize=[1, kernel_size, kernel_size, 1], strides=[1, stride, stride, 1],
						  padding=padding, name=name)

def avg_pooling(input_, kernel_size, stride, name, padding="SAME"):
	return tf.nn.avg_pool(input_, ksize=[1, kernel_size, kernel_size, 1], strides=[1, stride, stride, 1],
						  padding=padding, name=name)

def lrelu(x, leak=0.2, name="lrelu"):
	return tf.maximum(x, leak * x)

def relu(input_, name="relu"):
	return tf.nn.relu(input_, name=name)

def residule_block_33(input_, output_dim, kernel_size=3, stride=1, dilation=2, atrous=False, name="res"):
	if atrous:
		conv2dc0 = atrous_conv2d(input_=input_, output_dim=output_dim, kernel_size=kernel_size, dilation=dilation,
								 name=(name + '_c0'))
		conv2dc0_norm = batch_norm(input_=conv2dc0, name=(name + '_bn0'))
		conv2dc0_relu = relu(input_=conv2dc0_norm)
		conv2dc1 = atrous_conv2d(input_=conv2dc0_relu, output_dim=output_dim, kernel_size=kernel_size,
								 dilation=dilation, name=(name + '_c1'))
		conv2dc1_norm = batch_norm(input_=conv2dc1, name=(name + '_bn1'))
	else:
		conv2dc0 = conv2d(input_=input_, output_dim=output_dim, kernel_size=kernel_size, stride=stride,
						  name=(name + '_c0'))
		conv2dc0_norm = batch_norm(input_=conv2dc0, name=(name + '_bn0'))
		conv2dc0_relu = relu(input_=conv2dc0_norm)
		conv2dc1 = conv2d(input_=conv2dc0_relu, output_dim=output_dim, kernel_size=kernel_size, stride=stride,
						  name=(name + '_c1'))
		conv2dc1_norm = batch_norm(input_=conv2dc1, name=(name + '_bn1'))
	add_raw = input_ + conv2dc1_norm
	output = relu(input_=add_raw)
	return output

def copy_and_crop_and_merge(contract_layer, upsampling):
	contract_layer_shape = tf.shape(contract_layer)
	upsampling_shape = tf.shape(upsampling)
	contract_layer_crop = contract_layer
	return tf.concat(values=[contract_layer_crop, upsampling], axis=-1)
def leaky_relu(a, alpha=0.1):
	a = tf.maximum(alpha * a, a)
	return a


def generator(image, reuse=False, name="generator"):
	with tf.variable_scope(name):
		if reuse:
			tf.get_variable_scope().reuse_variables()
		else:
			assert tf.get_variable_scope().reuse is False
		conv1_weights = tf.get_variable('weights1', [3, 3,  1, 16],
											initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv1_biases = tf.get_variable('bias1', [16], initializer=tf.constant_initializer(0.0))
		conv1 = tf.nn.conv2d(image, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
		reglu1 = leaky_relu(tf.nn.bias_add(conv1, conv1_biases))
		conv1_1_weights = tf.get_variable('weights_1', [3, 3,  16, 16],
											initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv1_1_biases = tf.get_variable('bias_1', [16], initializer=tf.constant_initializer(0.0))
		conv1_1 = tf.nn.conv2d(reglu1, conv1_1_weights, strides=[1, 1, 1, 1], padding='SAME')
		reglu1_1 = leaky_relu(tf.nn.bias_add(conv1_1, conv1_1_biases))

		print(reglu1_1.get_shape())
		contract_layer1 = reglu1_1
		maxpool1 = tf.nn.max_pool(reglu1_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
		print(maxpool1.get_shape())

		conv2_weights = tf.get_variable('weights2', [3, 3, 16, 32],
											initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv2_biases = tf.get_variable('bias2', [32], initializer=tf.constant_initializer(0.0))
		conv2 = tf.nn.conv2d(maxpool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
		reglu2 = leaky_relu(tf.nn.bias_add(conv2, conv2_biases))

		conv2_1_weights = tf.get_variable('weights_2', [3, 3, 32, 32],
											initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv2_1_biases = tf.get_variable('bias_2', [32], initializer=tf.constant_initializer(0.0))
		conv2_1 = tf.nn.conv2d(reglu2, conv2_1_weights, strides=[1, 1, 1, 1], padding='SAME')
		reglu2_1 = leaky_relu(tf.nn.bias_add(conv2_1, conv2_1_biases))

		print(reglu2_1.get_shape())
		contract_layer2 = reglu2_1

		maxpool2 = tf.nn.max_pool(reglu2_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
		print(maxpool2.get_shape())

		conv3_weights = tf.get_variable('weights3', [3, 3, 32, 64],
											initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv3_biases = tf.get_variable('bias3', [64], initializer=tf.constant_initializer(0.0))
		conv3 = tf.nn.conv2d(maxpool2, conv3_weights, strides=[1, 1, 1, 1], padding='SAME')
		reglu3 = leaky_relu(tf.nn.bias_add(conv3, conv3_biases))

		conv3_1_weights = tf.get_variable('weights_3', [3, 3, 64, 64],
											initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv3_1_biases = tf.get_variable('bias_3', [64], initializer=tf.constant_initializer(0.0))
		conv3_1 = tf.nn.conv2d(reglu3, conv3_1_weights, strides=[1, 1, 1, 1], padding='SAME')
		reglu3_1 = leaky_relu(tf.nn.bias_add(conv3_1, conv3_1_biases))

		print(reglu3_1.get_shape())
		contract_layer3= reglu3_1
		maxpool3 = tf.nn.max_pool(reglu3_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
		print(maxpool3.get_shape())

		conv4_weights = tf.get_variable('weights4', [3, 3, 64, 128],
										initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv4_biases = tf.get_variable('bias4', [128], initializer=tf.constant_initializer(0.0))
		conv4 = tf.nn.conv2d(maxpool3, conv4_weights, strides=[1, 1, 1, 1], padding='SAME')
		reglu4 = leaky_relu(tf.nn.bias_add(conv4, conv4_biases))

		conv4_1_weights = tf.get_variable('weights_4', [3, 3, 128, 128],
										  initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv4_1_biases = tf.get_variable('bias_4', [128], initializer=tf.constant_initializer(0.0))
		conv4_1 = tf.nn.conv2d(reglu4, conv4_1_weights, strides=[1, 1, 1, 1], padding='SAME')
		reglu4_1 = leaky_relu(tf.nn.bias_add(conv4_1, conv4_1_biases))

		print(reglu4_1.get_shape())
		contract_layer4 = reglu4_1
		maxpool4 = tf.nn.max_pool(reglu4_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
		print(maxpool4.get_shape())

		conv5_weights = tf.get_variable('weights5', [3, 3, 128, 256],
											initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv5_biases = tf.get_variable('bias5', [256], initializer=tf.constant_initializer(0.0))
		conv5 = tf.nn.conv2d(maxpool4, conv5_weights, strides=[1, 1, 1, 1], padding='SAME')
		reglu5 = leaky_relu(tf.nn.bias_add(conv5, conv5_biases))

		conv5_1_weights = tf.get_variable('weights_5', [3, 3, 256, 256],
											initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv5_1_biases = tf.get_variable('bias_5', [256], initializer=tf.constant_initializer(0.0))
		conv5_1 = tf.nn.conv2d(reglu5, conv5_1_weights, strides=[1, 1, 1, 1], padding='SAME')
		reglu5_1 = leaky_relu(tf.nn.bias_add(conv5_1, conv5_1_biases))
		print(reglu5_1.get_shape())

		conv5_2_biases = tf.get_variable('bias_up1', [128], initializer=tf.constant_initializer(0.0))
		up_sample1 = tf.layers.conv2d_transpose(reglu5_1, 128, 2, strides=2, padding='SAME')
		reglu5_2 = leaky_relu(tf.nn.bias_add(up_sample1 , conv5_2_biases))
		print(reglu5_2.get_shape())

		merge1 = copy_and_crop_and_merge(contract_layer=contract_layer4, upsampling=reglu5_2)
		conv6_weights = tf.get_variable('weights6', [3, 3, 256, 128],
										initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv6_biases = tf.get_variable('bias6', [128], initializer=tf.constant_initializer(0.0))
		conv6 = tf.nn.conv2d(merge1, conv6_weights, strides=[1, 1, 1, 1], padding='SAME')
		reglu6 = leaky_relu(tf.nn.bias_add(conv6, conv6_biases))

		conv6_1_weights = tf.get_variable('weights_6', [3, 3, 128, 64],
										initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv6_1_biases = tf.get_variable('bias_6', [64], initializer=tf.constant_initializer(0.0))
		conv6_1 = tf.nn.conv2d(reglu6, conv6_1_weights, strides=[1, 1, 1, 1], padding='SAME')
		reglu6_1 = leaky_relu(tf.nn.bias_add(conv6_1, conv6_1_biases))
		print(reglu6_1.get_shape())

		conv6_2_biases = tf.get_variable('bias_up2', [64], initializer=tf.constant_initializer(0.0))
		up_sample2 = tf.layers.conv2d_transpose(reglu6_1, 64, 2, strides=2, padding='SAME')
		reglu6_2 = leaky_relu(tf.nn.bias_add(up_sample2 , conv6_2_biases))
		merge2 = copy_and_crop_and_merge(contract_layer=contract_layer3, upsampling=reglu6_2)
		conv7_weights = tf.get_variable('weights7', [3, 3, 128, 64],
											initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv7_biases = tf.get_variable('bias7', [64], initializer=tf.constant_initializer(0.0))
		conv7 = tf.nn.conv2d(merge2, conv7_weights, strides=[1, 1, 1, 1], padding='SAME')
		reglu7 = leaky_relu(tf.nn.bias_add(conv7, conv7_biases))

		conv7_1_weights = tf.get_variable('weights_7', [3, 3, 64, 32],
											initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv7_1_biases = tf.get_variable('bias_7', [32], initializer=tf.constant_initializer(0.0))
		conv7_1 = tf.nn.conv2d(reglu7, conv7_1_weights, strides=[1, 1, 1, 1], padding='SAME')
		reglu7_1 = leaky_relu(tf.nn.bias_add(conv7_1, conv7_1_biases))
		print(reglu7_1.get_shape())

		conv7_2_biases = tf.get_variable('bias_up3', [32], initializer=tf.constant_initializer(0.0))
		up_sample3 = tf.layers.conv2d_transpose(reglu7_1, 32, 2, strides=2, padding='SAME')
		reglu7_2 = leaky_relu(tf.nn.bias_add(up_sample3 , conv7_2_biases))
		print(reglu7_2.get_shape())


		merge3 = copy_and_crop_and_merge(contract_layer=contract_layer2, upsampling=reglu7_2)
		conv8_weights = tf.get_variable('weights8', [3, 3, 64, 32],initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv8_biases = tf.get_variable('bias8', [32], initializer=tf.constant_initializer(0.0))
		conv8 = tf.nn.conv2d(merge3, conv8_weights, strides=[1, 1, 1, 1], padding='SAME')
		reglu8 = leaky_relu(tf.nn.bias_add(conv8, conv8_biases))

		conv8_1_weights = tf.get_variable('weights_8', [3, 3, 32, 16],initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv8_1_biases = tf.get_variable('bias_8', [16], initializer=tf.constant_initializer(0.0))
		conv8_1 = tf.nn.conv2d(reglu8, conv8_1_weights, strides=[1, 1, 1, 1], padding='SAME')
		reglu8_1 = leaky_relu(tf.nn.bias_add(conv8_1, conv8_1_biases))
		print(reglu8_1.get_shape())

		conv8_2_biases = tf.get_variable('bias_up4', [16], initializer=tf.constant_initializer(0.0))
		up_sample4 = tf.layers.conv2d_transpose(reglu8_1, 16, 2, strides=2, padding='SAME')
		reglu8_2 = leaky_relu(tf.nn.bias_add(up_sample4 , conv8_2_biases))
		print(reglu8_2.get_shape())

		merge4 = copy_and_crop_and_merge(contract_layer=contract_layer1, upsampling=reglu8_2)
		conv9_weights = tf.get_variable('weights9', [3, 3, 32, 16],
										initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv9_biases = tf.get_variable('bias9', [16], initializer=tf.constant_initializer(0.0))
		conv9 = tf.nn.conv2d(merge4, conv9_weights, strides=[1, 1, 1, 1], padding='SAME')
		reglu9 = leaky_relu(tf.nn.bias_add(conv9, conv9_biases))

		conv9_1_weights = tf.get_variable('weights_9', [3, 3, 16, 1],
										  initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv9_1_biases = tf.get_variable('bias_9', [1], initializer=tf.constant_initializer(0.0))
		conv9_1 = tf.nn.conv2d(reglu9, conv9_1_weights, strides=[1, 1, 1, 1], padding='SAME')
		reglu9_1 = leaky_relu(tf.nn.bias_add(conv9_1, conv9_1_biases))
		print(reglu9_1.get_shape())

	return reglu9_1

def discriminator(image, df_dim=64, reuse=False, name="discriminator"):
	with tf.variable_scope(name):
		if reuse:
			tf.get_variable_scope().reuse_variables()
		else:
			assert tf.get_variable_scope().reuse is False
		h0 = lrelu(conv2d(input_=image, output_dim=df_dim, kernel_size=4, stride=2, name='d_h0_conv'))
		h1 = lrelu(
			batch_norm(conv2d(input_=h0, output_dim=df_dim * 2, kernel_size=4, stride=2, name='d_h1_conv'), 'd_bn1'))
		h2 = lrelu(
			batch_norm(conv2d(input_=h1, output_dim=df_dim * 4, kernel_size=4, stride=2, name='d_h2_conv'), 'd_bn2'))
		h3 = lrelu(
			batch_norm(conv2d(input_=h2, output_dim=df_dim * 8, kernel_size=4, stride=1, name='d_h3_conv'), 'd_bn3'))
		output = conv2d(input_=h3, output_dim=1, kernel_size=4, stride=1, name='d_h4_conv')
		return output
