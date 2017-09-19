import tensorflow as tf
import cv2

#图片翻转

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
    image_raw_data = tf.gfile.FastGFile("/Users/dudu/PycharmProjects/tensorflow/data/image/timg.jpg", "rb").read()

    # 按照jpeg的格式解码图片。
    image_data = tf.image.decode_jpeg(image_raw_data)

    # 显示原图片。
    cv2Show("Read by Tensorflow+Dispalyed by Opencv",image_data)

    # 上下翻转图像
    up_and_down = tf.image.flip_up_down(image_data)
    cv2Show("up and down",up_and_down)

    # 左右翻转图像
    left_and_right = tf.image.flip_left_right(image_data)
    cv2Show("left and right", left_and_right)

    # 沿对角线翻转图像
    transposed = tf.image.transpose_image(image_data)
    cv2Show("transposed image", transposed)

    # 以一定概率上下翻转图像
    random_up_and_down = tf.image.random_flip_up_down(image_data)
    cv2Show("random up and down", random_up_and_down)

    # 以一定概率左右翻转图像
    random_left_and_right = tf.image.random_flip_left_right(image_data)
    cv2Show("random left and right", random_left_and_right)

    cv2.waitKey()