#coding:utf-8

import tensorflow as tf
import numpy as np

embedding_matrix = np.array([[1, 1, 1, 1],
                            [2, 2, 2, 2],
                            [3, 3, 3, 3],
                            [4, 4, 4, 4],
                            [5, 5, 5, 5],
                            [0, 0, 0, 0]])
print(len(embedding_matrix))
last_index = len(embedding_matrix) - 1
input_index = [2, 1, 4, last_index]


embedding_vec = tf.nn.embedding_lookup(embedding_matrix, input_index)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(embedding_vec))






