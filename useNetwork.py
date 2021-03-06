import sys
import os
import numpy as np
import tensorflow as tf
from network import neural_network_model

x = tf.placeholder('float', [None, 280])
y = tf.placeholder('float')


def use_neural_network(input):
    prediction = neural_network_model(x)

    _ = [ord(_) for _ in input]
    input = np.pad(np.array(_), (0, 280 - len(_)), 'constant')
    input = np.array([input])

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        saver.restore(sess, os.path.dirname(os.path.abspath(__file__)) + '/model.ckpt')

        return sess.run(tf.argmax(prediction.eval(feed_dict={x: input}), 1)) + 1
