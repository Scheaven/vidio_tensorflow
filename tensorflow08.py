import tensorflow as tf

input1 = tf.placeholder(tf.int32)
input2 = tf.placeholder(tf.int32)

mul = tf.multiply(input1,input2)

with tf.Session() as sess:
    a = sess.run(mul,feed_dict={input1:7.,input2:3.})
    print(a)