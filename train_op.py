import tensorflow as tf
import numpy as np

_BETA_ = 0.5
_LEARNING_RATE_ = 2.0e-4

def train_opt(d_loss, d_vars):
	d_opt = tf.train.AdamOptimizer(_LEARNING_RATE_, beta1 = _BETA_).minimize(d_loss, var_list = d_vars)
	return d_opt
