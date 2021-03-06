import sys
import os
import numpy as np
import tensorflow as tf
import sqlite3
conn = sqlite3.connect(os.path.dirname(os.path.abspath(__file__)) + '/twitter-data-collector/database.sqlite')

NNodesHL1 = 500
NNodesHL2 = 500
NNodesHL3 = 500

c = conn.cursor()
hashtags = c.execute('SELECT * FROM hashtags').fetchall()
c.close()

n_classes = len(hashtags)


def neural_network_model(data):
    HL1 = {'weights': tf.Variable(tf.random_normal([280, NNodesHL1])),
           'biases': tf.Variable(tf.random_normal([NNodesHL1]))}

    HL2 = {'weights': tf.Variable(tf.random_normal([NNodesHL1, NNodesHL2])),
           'biases': tf.Variable(tf.random_normal([NNodesHL2]))}

    HL3 = {'weights': tf.Variable(tf.random_normal([NNodesHL2, NNodesHL3])),
           'biases': tf.Variable(tf.random_normal([NNodesHL3]))}

    OL = {'weights': tf.Variable(tf.random_normal([NNodesHL3, n_classes])),
          'biases': tf.Variable(tf.random_normal([n_classes])), }

    l1 = tf.add(tf.matmul(data, HL1['weights']), HL1['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, HL2['weights']), HL2['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, HL3['weights']), HL3['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.add(tf.matmul(l3, OL['weights']), OL['biases'])

    return output
