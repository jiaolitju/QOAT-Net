# ---从自己电脑磁盘文件中读取图片，存储到一个TFRecord文件中

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt   #导入python的库函数用于加载mat格式文件
from scipy.io import loadmat as load
import os
# import matplotlib.pyplot as plt

# 输入样本在本地磁盘中的地址
file_dir1 = "E:\\210401程序材料\\data\\pork\\p0\\"  #128_0505
file_dir2 = "E:\\210401程序材料\\data\\pork\\fai\\"
file_dir3 = "E:\\210401程序材料\\data\\pork\\ua\\"
file_dir4 = "E:\\210401程序材料\\data\\pork\\p0_1\\"

#输出TFRecord文件的地址
#filename="E:\\深度学习\\python程序\\Data\\trainTF\\input"

#创建一个writer来写TFRecord文件
writer=tf.python_io.TFRecordWriter("sys_train_data.tfrecords")

# 生成字符型的属性
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# 生成整数型的属性
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

#1368
for index in range(360):
    input_data =  load(os.path.join(file_dir1, '%d.mat' %(index+1) ))#(index+2768)
    input = input_data['p0']
    input_raw=np.reshape(input, [256, 256]) #返回一个"以文本方式表示"此对象的字符串
    # plt.imshow(input_raw)
    # # #plt.imshow(np.reshape(y_test,(128,128)))
    # plt.colorbar()
    # plt.show()
    # print(tf.shape(input_raw))
    # print(type(input_raw))
    input_raw=input_raw.tobytes()
    # print(type(input_raw))

    true_data1 = load(os.path.join(file_dir2, '%d.mat' % (index+1)))  # 每个图片的地址
    true1=true_data1['fai_1'] #np.array  dtype=np.float32  #miuasave   fai
    # print(true)
    true_raw1=np.reshape(true1, [256, 256])
    true_raw1=true_raw1.tobytes() #返回一个"以文本方式表示"此对象的字符串

    true_data2 = load(os.path.join(file_dir3, '%d.mat' % (index+1)))  # 每个图片的地址
    true2=true_data2['ua'] #np.array  dtype=np.float32  #_true
    # print(true)
    true_raw2=np.reshape(true2, [256, 256])
    true_raw2=true_raw2.tobytes() #返回一个"以文本方式表示"此对象的字符串

    true_data3 = load(os.path.join(file_dir4, '%d.mat' % (index+1)))  # 每个图片的地址
    true3 = true_data3['p0_1']  # np.array  dtype=np.float32  #miuasave   fai
    # print(true)
    true_raw3 = np.reshape(true3, [256, 256])
    true_raw3 = true_raw3.tobytes()  # 返回一个"以文本方式表示"此对象的字符串

    # 将一个样例转化成Example Protocol Buffer，并将所有的信息写入这个数据结构
    # tf.train下有Feature和Features，需要注意其区别,层级关系为Example->Features->Feature
    example = tf.train.Example(features=tf.train.Features(feature={
        'input_raw': _bytes_feature(input_raw),
        'true_raw1':_bytes_feature(true_raw1),
		'true_raw2': _bytes_feature(true_raw2),
		'true_raw3': _bytes_feature(true_raw3)}))

    # 将一个Example写入TFRecord文件中
    writer.write(example.SerializeToString())
writer.close()
