# ---从自己电脑磁盘文件中读取图片，存储到一个TFRecord文件中

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt   #导入python的库函数用于加载mat格式文件
from scipy.io import loadmat as load
import os
# import matplotlib。pyplot as plt

# 输入样本在本地磁盘中的地址
file_dir1 = "H:\\sys cyclegan\\数据\\12-09sy\\数据处理\\pmn\\"  #128_0505

#输出TFRecord文件的地址
#filename="E:\\深度学习\\python程序\\Data\\trainTF\\input"

#创建一个writer来写TFRecord文件
writer=tf.python_io.TFRecordWriter("test_data.tfrecords")

# 生成字符型的属性
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# 生成整数型的属性
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

#1368
for index in range(2880):
    input_data =  load(os.path.join(file_dir1, '%d.mat' % (index+1) ))
    input_raw = input_data['pmn']  #_true  gyp0  index jia 2768
    input_raw=np.reshape(input_raw, [256, 256]) #返回一个"以文本方式表示"此对象的字符串
    # plt.imshow(input_raw)
    # # #plt.imshow(np.reshape(y_test,(128,128)))
    # plt.colorbar()
    # plt.show()
    # print(input_raw)
    input_raw=input_raw.tobytes()
    # print(type(input_raw))


    # 将一个样例转化成Example Protocol Buffer，并将所有的信息写入这个数据结构
    # tf。train下有Feature和Features，需要注意其区别,层级关系为Example->Features->Feature
    example = tf.train.Example(features=tf.train.Features(feature={
        'input_raw': _bytes_feature(input_raw)}))

    # 将一个Example写入TFRecord文件中
    writer.write(example.SerializeToString())
writer.close()
