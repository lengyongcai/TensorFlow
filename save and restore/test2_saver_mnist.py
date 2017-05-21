#coding:utf-8

# import mnist data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./MNIST_data", one_hot=True)

import tensorflow as tf
import numpy as np

#parameters
learning_rate = 0.001
batch_size = 100
display_step = 1
model_path = "./checkpoint/model.ckpt"

# network parameters
n_hidden_1 = 256
n_hidden_2 = 256
n_input = 784
n_classes = 10

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])


# create model
def multilayer_preceptron(x, weights, biases):
    # hidden layer whit RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)

    # hidden layer whit RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)

    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# store layers weight & biases
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# construct model
pred = multilayer_preceptron(x, weights, biases)

# define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

init = tf.initialize_all_variables()

# saver op to save and restore all the variable
saver = tf.train.Saver()

# Running first session
print("Starting 1st seeeion ......")
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(3):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            #print("111", batch_x)
            #print("111_type:", type(batch_x))
            #print("111_shape:", np.shape(batch_x))

            # run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
            # compute average loss
            avg_cost += c/total_batch

        if epoch % display_step == 0:
            #print(sess.run(biases['out']))
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

    print("First Optimization Finished !!")



    # test model
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_pred, "float"))
    accuracy = accuracy.eval({x: mnist.test.images, y: mnist.test.labels})
    print("accuracy:", accuracy)

    # saver model weights to disk
    save_path = saver.save(sess, model_path)
    print("Model saved in file: %s" % save_path)


# Runing a new session
print("Starting 2nd session ......")

with tf.Session() as sess:
    sess.run(init)

    # restore model weights from previously saved model
    load_path = saver.restore(sess, model_path)
    print("Model restored from file: %s" % save_path)

    # resume training
    for epoch in range(7):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples/batch_size)

        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})

            avg_cost += c/total_batch

        if epoch % display_step == 0:
            # print(sess.run(biases['out']))
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))


    print("Secondary Optimization Finished !!")

    # test model
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_pred, "float"))
    accuracy = accuracy.eval({x: mnist.test.images, y: mnist.test.labels})
    print("accuracy:", accuracy)













