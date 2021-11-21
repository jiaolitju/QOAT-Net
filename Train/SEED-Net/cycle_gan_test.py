from net import *
# from net import *
from scipy.io import savemat
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from scipy.io import savemat
from scipy.io import loadmat as load

test_input_folder="H"
data_test=load(test_input_folder)
data_test = data_test['fai']
data_test = np.reshape(data_test, [1,256,256,1])

test_input_folder1="H"
data_test1=load(test_input_folder1)
data_test1 = data_test1['psy']
data_test1 = np.reshape(data_test1, [1,256,256,1])


#Parameters
lr=2e-4 #1e-2开始还行，后面溢出？ 1e-3从5降到0.5
batch_size=1
training_epochs=60
lamda=10.0  # L1 lamda
beta1=0.5  #momentum term of adam

X=tf.placeholder(tf.float32,[None,256, 256,1])
Y=tf.placeholder(tf.float32,[None,256, 256,1])

def l1_loss(src, dst):  # 定义l1_loss
	return tf.reduce_mean(tf.abs(src - dst))

def gan_loss(src, dst):  # 定义gan_loss，在这里用了二范数
	return tf.reduce_mean((src - dst) ** 2)

def main():
	fake_y = generator(image=X, reuse=False, name='generator_x2y')  # 得到生成的y域图像
	fake_x = generator(image=Y, reuse=False, name='generator_y2x')  # 得到生成的x域图像

	restore_var = [v for v in tf.global_variables() if 'generator' in v.name]  # 需要载入的已训练的模型参数
	saver = tf.train.Saver(var_list=restore_var, max_to_keep=1)  # 导入模型参数时使用
	# checkpoint = tf.train.latest_checkpoint("H:\\sys cyclegan")  # 读取模型参数

	with tf.Session() as sess:
		sess.run(tf.local_variables_initializer())
		# saver.restore(sess, checkpoint)  # 导入模型参数
		saver.restore(sess, "H:\\model.ckpt")  # 导入模型参数

		x_fake = sess.run(fake_x, feed_dict={Y: data_test1})
		x_fake = np.reshape(x_fake, (256, 256))
		# plt.imshow(np.reshape(x_fake, (256, 256)))
		# # plt.colorbar()
		# plt.show()
		# savemat("H:\\sys cyclegan\\测试结果暂存\\mnfake", {'xfake': x_fake})

		y_fake = sess.run(fake_y, feed_dict={X: data_test})
		y_fake = np.reshape(y_fake, (256, 256))
		plt.imshow(np.reshape(y_fake, (256, 256)))
		# plt.colorbar()
		plt.show()
		savemat("H:\\sys cyclegan\\测试结果暂存\\syfake", {'yfake': y_fake})





if __name__ == '__main__':
	main()