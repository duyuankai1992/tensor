import tensorflow as tf
import cv2

#图片的调色相

# 这里定义一个tensorflow读取的图片格式转换为opencv读取的图片格式的函数
# 请注意：
# 在tensorflow中，一个像素点的颜色顺序是R，G，B。
# 在opencv中，一个像素点的颜色顺序是B，G，R。
# 因此，我们循环遍历每一个像素点，将第0位的颜色和第2位的颜色数值换一下即可。
# 第一个参数name：将要显示的窗口名称。
# 第二个参数image：储存图片信息的一个tensor。
def cv2Show(name="", image=None):
    # 获取矩阵信息
    np = image.eval()
    # 获取行数列数
    row, col = len(np),len(np[1])

    # 两重循环遍历
    for i in range(row):
        for j in range(col):
            # 交换数值
            tmp = np[i][j][0]
            np[i][j][0] = np[i][j][2]
            np[i][j][2] = tmp

    # 显示图片
    cv2.imshow(name,np)
    pass

# tensorflow会话
with tf.Session() as sess:
    # 以二进制的方式读取图片。
    image_raw_data = tf.gfile.FastGFile("/Users/dudu/PycharmProjects/tensor/timg.jpg", "rb").read()

    # 按照jpeg的格式解码图片。
    image_data = tf.image.decode_jpeg(image_raw_data)

    # 显示原图片。
    cv2Show("Read by Tensorflow+Dispalyed by Opencv",image_data)

    # adjust_hue(image, delta, name=None)
    # delte的范围:[-1,1]
    # 将图片的色相+0.1
    adjusted1 = tf.image.adjust_hue(image_data, 0.1)
    cv2Show("hue +0.1", adjusted1)

    # 将图片的色相+0.3
    adjusted2 = tf.image.adjust_hue(image_data, 0.3)
    cv2Show("hue +0.3", adjusted2)

    # 将图片的色相+0.6
    adjusted3 = tf.image.adjust_hue(image_data, 0.6)
    cv2Show("hue +0.6", adjusted3)

    # 将图片的色相+0.9
    adjusted4 = tf.image.adjust_hue(image_data, 0.9)
    cv2Show("hue +0.9", adjusted4)

    # 随机调整图像的色相：
    # random_hue(image, max_delta, seed=None)
    # image：待调整的图像
    # max_delta：在[-max_delta,max_delta]的范围随机调整图像的色相。max_delta的范围[0,0.5]。
    # seed：随机数种子
    adjusted5 = tf.image.random_hue(image_data,0.4)
    cv2Show("random hue", adjusted5)

    cv2.waitKey()