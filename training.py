import sys
import numpy as np
import tensorflow as tf
import sqlite3
conn = sqlite3.connect('twitter-data-collector/database.sqlite')

NNodesHL1 = 500
NNodesHL2 = 500
NNodesHL3 = 500

NEpochs = 1000

x = tf.placeholder('float', [None, 280])
y = tf.placeholder('float')

c = conn.cursor()
hashtags = c.execute('SELECT * FROM hashtags').fetchall()
data = c.execute('SELECT * FROM tweets')

dataset = []

for tweet in data.fetchall():
    _ = [ord(_) for _ in tweet[2]]
    tweet = (tweet[1], np.pad(np.array(_), (0, 280 - len(_)), 'constant'))
    dataset.append(tweet)

c.close()

n_classes = len(hashtags)

newDataset = {
    'tweets': [],
    'sentiment': []
}
for data in dataset:
    newDataset['tweets'].append(data[1])
    sentiment = np.zeros(n_classes)
    sentiment[data[0] - 1] = 1
    newDataset['sentiment'].append(sentiment)
dataset = newDataset


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


def train_neural_network(x):
    prediction = neural_network_model(x)
    print prediction
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(NEpochs):
            epoch_loss = 0
            _, c = sess.run([optimizer, cost], feed_dict={x: dataset['tweets'], y: dataset['sentiment']})
            epoch_loss += c

            print('Epoch', epoch, 'completed out of', NEpochs, 'loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x: dataset['tweets'], y: dataset['sentiment']}))

        print 'Saved model in ' + saver.save(sess, './model.ckpt')


train_neural_network(x)
