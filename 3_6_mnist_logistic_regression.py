""" Часть I. Глава 3. Раздел 3.6. Распознавание рукописных цифр на TensorFlow"""

import tensorflow as tf

# загружаем данные MNIST
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# создаем заглушки, [None, 784] означает произвольное число 784-мерных векторов
x = tf.placeholder(tf.float32, [None, 784])

# задаем переменные для обучения
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# задаем модель
y = tf.nn.softmax(tf.matmul(x, W) + b)

# создаем функцию потерь
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), axis=[1]))

# задаем оптимизатор
train_step = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(cross_entropy)

# запускаем сессию
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy: %s" % sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
