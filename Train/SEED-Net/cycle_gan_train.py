from net import *
from scipy.io import savemat
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from scipy.io import savemat
from scipy.io import loadmat as load

#Parameters

lr=2e-4
batch_size=1
training_epochs=70
lamda=10.0  # L1 lamda
beta1=0.5  #momentum term of adam

X=tf.placeholder(tf.float32,[None,256, 256,1])
Y=tf.placeholder(tf.float32,[None,256, 256,1])

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
										   'A_raw': tf.FixedLenFeature([], tf.string),
										   'B_raw': tf.FixedLenFeature([], tf.string),
									   })
	# 得到原始数据、尺寸。
	A_data1=features['A_raw'] #
	B_data1 = features['B_raw']


	#tf.decode_raw可以将字符串解析成图像对应的像素数组
	decode_A = tf.decode_raw(A_data1, tf.float32)
	decode_A = tf.reshape(decode_A, [256, 256, 1])
	# decode_input=tf.transpose(decode_input)
	# decode_input=tf.reshape(decode_input, [128, 128,1])

	decode_B = tf.decode_raw(B_data1, tf.float64)  #当用miua时是float64,  fai是float32
	decode_B = tf.reshape(decode_B, [256, 256, 1])
	# decode_label=tf.transpose(decode_label)
	# decode_label=tf.reshape(decode_label, [128, 128,1])


	# 进行批处理
	x_batch, y_batch = tf.train.shuffle_batch(([decode_A,decode_B]), batch_size=batchsize,
											  num_threads=1, capacity=300,
											  min_after_dequeue=200) #
	return x_batch, y_batch

def l1_loss(src, dst):  # 定义l1_loss
	return tf.reduce_mean(tf.abs(src - dst))

def gan_loss(src, dst):  # 定义gan_loss，在这里用了二范数
	return tf.reduce_mean((src - dst) ** 2)

def main():
	fake_y = generator(image=X, reuse=False, name='generator_x2y')  # 生成的y域图像
	fake_x_ = generator(image=fake_y, reuse=False, name='generator_y2x')  # 重建的x域图像
	fake_x = generator(image=Y, reuse=True, name='generator_y2x')  # 生成的x域图像
	fake_y_ = generator(image=fake_x, reuse=True, name='generator_x2y')  # 重建的y域图像
	##identify loss 部分
	idl_y = generator(image=Y, reuse=True, name='generator_x2y')
	idl_x = generator(image=X, reuse=True, name='generator_y2x')

	dy_fake = discriminator(image=fake_y, reuse=False, name='discriminator_y')  # 判别器返回的对生成的y域图像的判别结果
	dx_fake = discriminator(image=fake_x, reuse=False, name='discriminator_x')  # 判别器返回的对生成的x域图像的判别结果
	dy_real = discriminator(image=Y, reuse=True, name='discriminator_y')  # 判别器返回的对真实的y域图像的判别结果
	dx_real = discriminator(image=X, reuse=True, name='discriminator_x')  # 判别器返回的对真实的x域图像的判别结果

	##identify loss 部分
	idloss=l1_loss(idl_x,X)+l1_loss(idl_y, Y)

	gen_loss = gan_loss(dy_fake, tf.ones_like(dy_fake)) + gan_loss(dx_fake,tf.ones_like(dx_fake)) \
			   + lamda * l1_loss(X, fake_x_) + lamda * l1_loss(Y, fake_y_) + idloss # 计算生成器的loss

	dy_loss_real = gan_loss(dy_real, tf.ones_like(dy_real))  # 计算判别器判别的真实的y域图像的loss
	dy_loss_fake = gan_loss(dy_fake, tf.zeros_like(dy_fake))  # 计算判别器判别的生成的y域图像的loss
	dy_loss = (dy_loss_real + dy_loss_fake) / 2  # 计算判别器判别的y域图像的loss

	dx_loss_real = gan_loss(dx_real, tf.ones_like(dx_real))  # 计算判别器判别的真实的x域图像的loss
	dx_loss_fake = gan_loss(dx_fake, tf.zeros_like(dx_fake))  # 计算判别器判别的生成的x域图像的loss
	dx_loss = (dx_loss_real + dx_loss_fake) / 2  # 计算判别器判别的x域图像的loss

	dis_loss = dy_loss + dx_loss  #计算判别器的loss

	g_vars = [v for v in tf.trainable_variables() if 'generator' in v.name]  # 所有生成器的可训练参数
	d_vars = [v for v in tf.trainable_variables() if 'discriminator' in v.name]  # 所有判别器的可训练参数

	# lr = tf.placeholder(tf.float32, None, name='learning_rate')  # 训练中的学习率
	d_optim = tf.train.AdamOptimizer(lr, beta1=beta1)  # 判别器训练器
	g_optim = tf.train.AdamOptimizer(lr, beta1=beta1)  # 生成器训练器

	d_grads_and_vars = d_optim.compute_gradients(dis_loss, var_list=d_vars)  # 计算判别器参数梯度
	d_train = d_optim.apply_gradients(d_grads_and_vars)  # 更新判别器参数
	g_grads_and_vars = g_optim.compute_gradients(gen_loss, var_list=g_vars)  # 计算生成器参数梯度
	g_train = g_optim.apply_gradients(g_grads_and_vars)  # 更新生成器参数

	train_op = tf.group(d_train, g_train)  # train_op表示了参数更新操作
	# config = tf.ConfigProto()
	# config.gpu_options.allow_growth = True  # 设定显存不超量使用
	# sess = tf.Session(config=config)  # 新建会话层

	x_batch,  y_batch = get_Batch("train_data.tfrecords", batch_size, training_epochs)
	saver = tf.train.Saver()

	costg=[]
	costd=[]
	with tf.Session() as sess:
		# Initilizaing the variables
		sess.run(tf.global_variables_initializer())
		sess.run(tf.local_variables_initializer())
		# saver.restore(sess, "D:\\Chentting\\qPAT\\zaijiagongzuo结果\\p1f100u200-1890个数据-直接训练\\model.ckpt")
		#开启协调器
		coord = tf.train.Coordinator()
		# 使用start_queue_runners 启动内存队列填充
		threads = tf.train.start_queue_runners(sess=sess, coord=coord)
		ites=0
		try:
			while not coord.should_stop():
				x_data,y_data=sess.run([x_batch, y_batch]) #在会话中取出image和label
				feed_dict = {X: x_data, Y: y_data}  # 得到feed_dict
				gen_loss_value, dis_loss_value, _ = sess.run([gen_loss, dis_loss, train_op],
															 feed_dict=feed_dict)  # 得到每个step中的生成器和判别器loss
				# if ites%870==0:
				print("After %d training epoch(s), G loss is %g, D loss is %g" % (ites,gen_loss_value, dis_loss_value))
				if ites%3240==0:
					costg.append(gen_loss_value)
					costd.append(dis_loss_value)
				if ites%64800==0:
					saver.save(sess, "E:\\wc\\师姐留资料\\风格迁移\\cycle-GAN\\模型\\model.ckpt", global_step=ites)
				ites=ites+1
		except tf.errors.OutOfRangeError:
			print("---Train end---")
		finally:
			# 协调器coord发出所有线程终止信号
			coord.request_stop()
		# 把开启的线程加入主线程，等待threads结束
		coord.join(threads)
		#####保存模型
		save_path = saver.save(sess, "E:\\wc\\师姐留资料\\风格迁移\\cycle-GAN\\模型\\model.ckpt")  # 保存会话中的所有内容？只保存参数模型 E:\wc\师姐留资料\风格迁移\cycle-GAN\模型  E:\\wc\\师姐留资料\\风格迁移\\cycle-GAN\\模型\\model.ckpt

		savemat("E:\\wc\\师姐留资料\\风格迁移\\cycle-GAN\\模型\\costg", {'costg': costg})
		savemat("E:\\wc\\师姐留资料\\风格迁移\\cycle-GAN\\模型\\costd", {'costd': costd})


if __name__ == '__main__':
	main()