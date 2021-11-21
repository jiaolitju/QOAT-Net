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

test_input_folder="G:\\QOAT-Net程序代码\\SEED-Net\\phantom result\\input simulation p0.mat"
data_test=load(test_input_folder)
data_test = data_test['p0']
data_test = np.reshape(data_test, [1,256,256,1])

test_input_folder1="G:\\QOAT-Net程序代码\\SEED-Net\\phantom result\\input simulation p0.mat"
data_test1=load(test_input_folder1)
data_test1 = data_test1['p0']
data_test1 = np.reshape(data_test1, [1,256,256,1])



lr=2e-4
batch_size=1
training_epochs=60
lamda=10.0
beta1=0.5

X=tf.placeholder(tf.float32,[None,256, 256,1])
Y=tf.placeholder(tf.float32,[None,256, 256,1])

def l1_loss(src, dst):
	return tf.reduce_mean(tf.abs(src - dst))

def gan_loss(src, dst):
	return tf.reduce_mean((src - dst) ** 2)

def main():
	fake_y = generator(image=X, reuse=False, name='generator_x2y')
	fake_x = generator(image=Y, reuse=False, name='generator_y2x')

	restore_var = [v for v in tf.global_variables() if 'generator' in v.name]
	saver = tf.train.Saver(var_list=restore_var, max_to_keep=1)

	with tf.Session() as sess:
		sess.run(tf.local_variables_initializer())
		saver.restore(sess, "G:\\QOAT-Net程序代码\\SEED-Net\\model\\model.ckpt")

		x_fake = sess.run(fake_x, feed_dict={Y: data_test1})
		x_fake = np.reshape(x_fake, (256, 256))

		y_fake = sess.run(fake_y, feed_dict={X: data_test})
		y_fake = np.reshape(y_fake, (256, 256))

		plt.imshow(np.reshape(y_fake, (256, 256)))
		# plt.colorbar()
		plt.show()
		#savemat("E:\\210401程序材料\\save_test\\syfake", {'yfake': y_fake})





if __name__ == '__main__':
	main()