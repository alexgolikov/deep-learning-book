""" Часть II. Глава 3. Раздел 5.3. Свертки для распознавания цифр"""

import tensorflow as tf

# загружаем данные MNIST
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# создаем заглушки, [None, 784] означает произвольное число 784-мерных векторов
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# переформатируем входной вектор в виде двумерного массива 28х28 (-1 - неизвестный размер мини-батча, 1 - одно число для каждого пикселя)
x_image = tf.reshape(x, [-1, 28, 28, 1])

# создаем сверточный слой (5х5 - размер ядра свертки, 32 - число фильтров, 1 - число цветовых каналов)
W_conv_1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
b_conv_1 = tf.Variable(tf.constant(0.1, shape=[32]))
conv_1 = tf.nn.conv2d(x_image, W_conv_1, strides=[1, 1, 1, 1], padding="SAME") + b_conv_1

# задаем функцию активации
h_conv_1 = tf.nn.relu(conv_1)

# слой субдискретизаци
h_pool_1 = tf.nn.max_pool(h_conv_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

# добавляем еще один слой по аналогии с первым, но с 64 фильтрами
W_conv_2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
b_conv_2 = tf.Variable(tf.constant(0.1, shape=[64]))
conv_2 = tf.nn.conv2d(h_pool_1, W_conv_2, strides=[1, 1, 1, 1], padding="SAME") + b_conv_2
h_conv_2 = tf.nn.relu(conv_2)
h_pool_2 = tf.nn.max_pool(h_conv_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

# делаем плоский слой из двумерного
h_pool_2_flat = tf.reshape(h_pool_2, [-1, 7*7*64])

# добавляем полносвязный слой на 1024 нейрона
W_fc_1 = tf.Variable(tf.truncated_normal([7*7*64, 1024], stddev=0.1))
b_fc_1 = tf.Variable(tf.constant(0.1, shape=[1024]))
h_fc_1 = tf.nn.relu(tf.matmul(h_pool_2_flat, W_fc_1) + b_fc_1)

# регуляризуем дропаутом
keep_probability = tf.placeholder(tf.float32)
h_fc_1_drop = tf.nn.dropout(h_fc_1, keep_probability)

# добавляем второй полносвязный слой на 10 выходов
W_fc_2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1))
b_fc_2 = tf.Variable(tf.constant(0.1, shape=[10]))
logit_conv = tf.matmul(h_fc_1_drop, W_fc_2) + b_fc_2
y_conv = tf.nn.softmax(logit_conv)

# определяем ошибку и оптимизатор
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit_conv, labels=y))
train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# запускаем сессию
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for i in range(10000):
    batch_xs, batch_ys = mnist.train.next_batch(64)
    sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_probability: 0.5})

# выводим результат
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_probability: 1.}))
