# from net_my import *
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

batch_size=1
training_epochs=1

X=tf.placeholder(tf.float32,[None,256, 256,1])

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
									   })
	# 得到原始数据、尺寸。
	input_data1=features['input_raw'] #

	#tf.decode_raw可以将字符串解析成图像对应的像素数组
	decode_input = tf.decode_raw(input_data1, tf.float32)
	decode_input = tf.reshape(decode_input, [256, 256, 1])
	# print(decode_input.get_shape())

	# 进行批处理
	x_batch = tf.train.batch(([decode_input]), batch_size=batchsize,
											  num_threads=1, capacity=300) #
	return x_batch

def main():
	fake_y = generator(image=X, reuse=False, name='generator_x2y')  # 生成的y域图像

	restore_var = [v for v in tf.global_variables() if 'generator' in v.name]  # 需要载入的已训练的模型参数
	saver = tf.train.Saver(var_list=restore_var, max_to_keep=1)  # 导入模型参数时使用
	# checkpoint = tf.train.latest_checkpoint("H:\\sys cyclegan")  # 读取模型参数

	x_batch = get_Batch("test_data.tfrecords", batch_size, training_epochs)
	saver = tf.train.Saver()

	with tf.Session() as sess:
		# Initilizaing the variables
		sess.run(tf.local_variables_initializer())
		# saver.restore(sess, checkpoint)  # 导入模型参数
		saver.restore(sess, "H:\\sys cyclegan\\save\\2020-05-17\\1209 labetrain bs1 gunet idloss\\model.ckpt")
		#开启协调器
		coord = tf.train.Coordinator()
		# 使用start_queue_runners 启动内存队列填充
		threads = tf.train.start_queue_runners(sess=sess, coord=coord)
		ites=1
		try:
			while not coord.should_stop():
				x_data=sess.run([x_batch]) #在会话中取出image和label
				x_data = np.reshape(x_data, [batch_size, 256, 256, 1])
				# plt.imshow(np.reshape(data, (256, 256)))
				# # plt.colorbar()
				# plt.show()
				feed_dict = {X: x_data}  # 得到feed_dict
				yfake = sess.run([fake_y ],feed_dict=feed_dict)  # 得到每个step中的生成器和判别器loss
				yfake = np.reshape(yfake, (256, 256))
				savemat(os.path.join("H:\\sys cyclegan\\数据\\cyclegan实验数据\\2020-05-17-1209 fsy数据处理\\p0\\" '%d' % (ites)), {'fpsy': yfake})
				ites = ites + 1
		except tf.errors.OutOfRangeError:
			print("---Train end---")
		finally:
			# 协调器coord发出所有线程终止信号
			coord.request_stop()
		# 把开启的线程加入主线程，等待threads结束
		coord.join(threads)



if __name__ == '__main__':
	main()