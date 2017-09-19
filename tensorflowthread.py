import tensorflow as tf

# 创建一个先进先出队列，指定队列中最多可以保存两个元素，并指定数据类型为整形。
q = tf.FIFOQueue(capacity=2, dtypes=tf.int32, name="queue")

# 使用enqueue_many函数来初始化队列中的元素。
# 和变量初始化类似，在使用队列之前需要明确调用这个初始化过程。
init = q.enqueue_many(([0, 10],))

# 使用dequeue函数将队列中的第一个元素出队列。这个元素的值将被保存在变量x中。
x = q.dequeue()

# 将得到的值加1
y = x + 1

# 将加1后的值再重新加入到队列。
q_inc = q.enqueue(y)

# Tensorflow会话
with tf.Session() as sess:

    # 运行初始化队列的操作。
    init.run()
    for i in range(5):

        # 运行q_inc将执行数据出队列，出队的元素值加1，重新加入队列的整个过程。
        v, _ = sess.run([x, q_inc])

        # 打印出队元素的值。
        print(v)