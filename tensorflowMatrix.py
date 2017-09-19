import tensorflow as tf

# Tensorflow交互式会话
tf.InteractiveSession()

# 定义5x5大小的一个矩阵变量
a = tf.Variable(tf.truncated_normal(shape=[5, 5], dtype=tf.float32))

# 进行切片操作，起始位置为[1,1]（从0开始），大小[2,2]
b = tf.slice(a, [1, 1], [2, 2])

# 同上
c = tf.Variable(tf.truncated_normal(shape=[2, 6, 5], dtype=tf.float32))

d = tf.slice(c, [0, 2, 3], [2, 3, 1])

# 全局变量初始化
tf.global_variables_initializer().run()

# 输出
print("Example 01")

print("the original matrix:\n", a.eval())

print("after being sliced:\n", b.eval())

print("Example 02")

print("the original matrix:\n", c.eval())

print("after being sliced:\n", d.eval())