import tensorflow as tf
import numpy as np

input = tf.placeholder(dtype=tf.float32,shape=[5,5,3])
filter = tf.constant(value=1, shape=[3,3,3,5], dtype=tf.float32)
conv0 = tf.nn.atrous_conv2d(input,filters=filter,rate=2,padding='VALID')

with tf.Session() as sess:
    img = np.array([3,5,5,3])
    out = sess.run(conv0,feed_dict={input:img})
    print(out.shape)

