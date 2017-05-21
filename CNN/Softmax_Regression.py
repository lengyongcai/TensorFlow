#coding:utf-8
""" 
Created on 2016-07-17 @author: yongcai
使用softmax_regression　方法来预测ＭＮＩＳＴ数据集的图片识别
"""


import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)

print("train images:", mnist.train.images.shape, mnist.train.labels.shape)
print("test images :", mnist.test.images.shape, mnist.test.labels.shape)
print("valid images:", mnist.validation.images.shape, mnist.validation.images.shape)



x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y)))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)




sess = tf.InteractiveSession()

init = tf.initialize_all_variables()
sess.run(init)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})



correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))



