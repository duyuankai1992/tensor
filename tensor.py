import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time

mnist = input_data.read_data_sets('/Users/dudu/PycharmProjects/tensorflow/data/mnist',one_hot=True)

batch_size = 128
X = tf.placeholder(tf.float32,[batch_size,784],name='X_placeholder')
Y = tf.placeholder(tf.int16,[batch_size,10],name='Y_placeholder')

W = tf.Variable(tf.random_normal(shape=[784,10],stddev=0.01),name='weight')
b = tf.Variable(tf.zeros([1,10]),name='bias')

logits = tf.add(tf.matmul(X,W),b)
entropy = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = Y, name = 'entropy')
loss = tf.reduce_mean(entropy)

learning_rate = 0.03
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

n_epochs = 30
with tf.Session() as sess:
    writer = tf.summary.FileWriter('./graphs/logistic_reg2',sess.graph)
    start_time = time.time()
    sess.run(tf.global_variables_initializer())
    n_batchs = int(mnist.train.num_examples/batch_size)
    for i in range(n_epochs):
        total_loss = 0
        for _ in range (n_batchs):
            X_batch,Y_batch = mnist.train.next_batch(batch_size)
            _, loss_batch = sess.run([optimizer, loss],feed_dict={X:X_batch,Y:Y_batch})
            total_loss += loss_batch
        print('Average loss in a batch (Epoch {0}): {1}'.format(i,total_loss/n_batchs))
    print('Total time: {0} seconds '.format(time.time() - start_time))
    print('Optimization Done!')

    preds = tf.nn.softmax(logits)
    isCorrect_preds = tf.equal(tf.argmax(preds,1),tf.argmax(Y,1))
    accuracy = tf.reduce_sum(tf.cast(isCorrect_preds,tf.float32))

    n_batchs = int(mnist.test.num_examples/batch_size)
    total_correct_preds = 0
    for i in range(n_batchs):
        X_batch,Y_batch = mnist.test.next_batch(batch_size)
        accuracy_batch = sess.run([accuracy],feed_dict={X:X_batch,Y:Y_batch})
        total_correct_preds += accuracy_batch[0]
    print('Accuracy: {0}'.format(total_correct_preds/mnist.test.num_examples))
    writer.close()