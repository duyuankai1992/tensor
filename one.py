#-- coding: UTF-8 --
'''
session的通用使用方法
'''
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mlpimg


filename = '/Users/dudu/PycharmProjects/tensorflow/data/image/u=1670324097,1966289191&fm=27&gp=0.jpg'
raw_image = mlpimg.imread(filename)
image = tf.placeholder('uint8',[None, None, 3])

slice1 = tf.slice(image,[0,0,0],[1000,1000,-1])
slice2 = tf.slice(image,[1001,1001,0],[-1,-1,-1])
with tf.Session() as session:
    y1 = session.run(slice1,feed_dict={image:raw_image})
    y2 = session.run(slice2,feed_dict={image:raw_image})

#注意，如果要进行显示图像，则必须先inshow(tensor)一下。
plt.imshow(y2)
plt.show()
print('OK!')

#解析placeholder
"""
在tensorflow中，就是先将各种操作定义好。比如即使是一个网络，不也是各种操作的集合吗？
对于一些变量，运行时才统一赋予初值（喂入数据），这些变量除了简单的constant，Variable之外
还有一般要用到placeholder。placeholder从字面意思上看就是占位符，和动态分配的list类似啊。
功能强大，可以设置该变量的类型，数据的维度之类的。比如tf.placeholder('uint8',[None,None,3])
就可以处理任何3个通道的图片。
"""
#其他解析
"""
1. tf.slice(x,[start1,start2,...startn],[end1,end2,...endn])
2. 这里要牢牢记住的就是：先搭框架！搭好之后再统一喂入数据。比如
    slice1 = tf.slice(image,[0,0,0],[1000,1000,-1])时候
"""