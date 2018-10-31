import tensorflow as tf
import numpy as np

x_data = np.random.randn(100).astype(np.float32)
y_data = x_data*0.1 + 0.3

Weights = tf.Variable(tf.random_uniform([1],-1.0,1.0))
biases = tf.Variable(tf.zeros([1]))
y = Weights*x_data + biases

loss = tf.reduce_mean(tf.square(y-y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    for step in range(20):
        sess.run(train)
        # if step%20 == 0:
        print(step,sess.run(Weights),sess.run(biases))