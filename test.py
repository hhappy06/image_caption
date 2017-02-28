import numpy as np
import tensorflow as tf


a = tf.constant(range(9), shape = [3, 3])
b = tf.constant(range(9,18), shape = [3, 3])
res = tf.pack([a, b], 1)
print res.get_shape()
ff = res[:,0,:]

with tf.Session() as session:
	ff_ = session.run(ff)
	print ff_

	