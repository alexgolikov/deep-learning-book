""" Часть I. Глава 3. Раздел 3.6. Распознавание рукописных цифр на TensorFlow"""

import tensorflow as tf

# загружаем данные MNIST
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# создаем заглушки, [None, 784] означает произвольное число 784-мерных векторов
x = tf.placeholder(tf.float32, [None, 784])

# слой ReLU
W_relu = tf.Variable(tf.truncated_normal([784, 100], stddev=0.1))
b_relu = tf.Variable(tf.truncated_normal([100], stddev=0.1))

# скрытый слой
h = tf.nn.relu(tf.matmul(x, W_relu) + b_relu)

# слой дропаута
keep_probability = tf.placeholder(tf.float32)
h_drop = tf.nn.dropout(h, keep_probability)

# softmax-слой
W = tf.Variable(tf.zeros([100, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(h_drop, W) + b)

# создаем функцию потерь
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), axis=[1]))

# задаем оптимизатор
train_step = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(cross_entropy)

# запускаем сессию
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_probability: 0.5})

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy: %s" % sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_probability: 1.}))
