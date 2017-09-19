import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

n_examples = 300
xs = np.linspace(-3, 3, n_examples)
ys = np.sin(xs) + np.random.uniform(-0.5, 0.5, n_examples)

X = tf.placeholder(tf.float32, name='X')
Y = tf.placeholder(tf.float32, name='Y')
W = tf.Variable(tf.random_normal([1]), name='W')
W_2 = tf.Variable(tf.random_normal([1], name='W_2'))
W_3 = tf.Variable(tf.random_normal([1]), name='W_3')
b = tf.Variable(tf.random_normal([1]), name='b')

y_ = tf.add(tf.multiply(X,W),b)
y_ = tf.add(y_, tf.multiply(tf.pow(X,2),W_2))
y_ = tf.add(y_, tf.multiply(tf.pow(X,3),W_3))

n_samples = xs.shape[0]
loss = tf.reduce_sum(tf.square(Y-y_))/n_samples

learning_rate = 0.03
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

with tf.Session() as sess:
    writer = tf.summary.FileWriter('./graphs/polynomial_reg',sess.graph)
    sess.run(tf.global_variables_initializer())
    for i in range (201):
        total_loss = 0
        for x,y in zip(xs, ys):
            _,l = sess.run([optimizer, loss],feed_dict={X:x, Y:y})
            total_loss += l
        if i%20 == 0:
            print('Epoch{0}:{1}'.format(i,total_loss/n_samples))
    writer.close()
    W,W_2,W_3,b = sess.run([W,W_2,W_3,b])


plt.plot(xs,ys,'bo',label='Real Data')
plt.plot(xs,xs*W+np.power(xs,2)*W_2+np.power(xs,3)*W_3+b,'r-',lw=5,label='Predicted Data')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Polynomial Regression')
plt.show()