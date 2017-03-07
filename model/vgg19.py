from __future__ import absolute_import
import os
import numpy as np
import tensorflow as tf

_VGG19_IMAGE_MEAN = [103.939, 116.779, 123.68]
_WEIGHT_INDEX = 0
_BIAS_INDEX = 1
_REGULAR_FACTOR = 1.0e-4
_LEARNING_RATE = 1.0e-4

_VGG19_NETWORK = {
	'conv1_1': [3, 64],
	'conv1_2': [3, 64],
	'conv2_1': [3, 128],
	'conv2_2': [3, 128],
	'conv3_1': [3, 256],
	'conv3_2': [3, 256],
	'conv3_3': [3, 256],
	'conv3_4': [3, 256],
	'conv4_1': [3, 512],
	'conv4_2': [3, 512],
	'conv4_3': [3, 512],
	'conv4_4': [3, 512],
	'conv5_1': [3, 512],
	'conv5_2': [3, 512],
	'conv5_3': [3, 512],
	'conv5_4': [3, 512],
	'fc6': [4096],
	'fc7': [4096],
	'fc8': [1000],
}

_CONV_KERNEL_STRIDES = [1, 1, 1, 1]
_MAX_POOL_KSIZE = [1, 2, 2, 1]
_MAX_POOL_STRIDES = [1, 2, 2, 1]

class VGG19:
	def __init__(self, name = 'vgg19'):
		self.name = name

	def inference(self, input_images, tensor_trainable, initialized_parameter_file = None):
		if initialized_parameter_file and os.path.exists(initialized_parameter_file):
			self.initialized_parameter_dict = np.load(initialized_parameter_file, encoding = 'latin1').item()
		else:
			self.initialized_parameter_dict = None

		# input_images is a placeholder with [None, height, width, nchannels]
		r, g, b = tf.split(input_images, 3, 3)
		# check the image size
		whiten_images = tf.concat([
			b - _VGG19_IMAGE_MEAN[0],
			g - _VGG19_IMAGE_MEAN[1],
			r - _VGG19_IMAGE_MEAN[2]], 3)

		with tf.variable_scope(self.name):
			# construct VGG19 network -- convolution layer
			conv1_1 = self._construct_conv_layer(input_images, 'conv1_1')
			conv1_2 = self._construct_conv_layer(conv1_1, 'conv1_2')
			pool1 = self._max_pool(conv1_2, 'pool1')

			conv2_1 = self._construct_conv_layer(pool1, 'conv2_1')
			conv2_2 = self._construct_conv_layer(conv2_1, 'conv2_2')
			pool2 = self._max_pool(conv2_2, 'pool2')

			conv3_1 = self._construct_conv_layer(pool2, 'conv3_1')
			conv3_2 = self._construct_conv_layer(conv3_1, 'conv3_2')
			conv3_3 = self._construct_conv_layer(conv3_2, 'conv3_3')
			conv3_4 = self._construct_conv_layer(conv3_3, 'conv3_4')
			pool3 = self._max_pool(conv3_4, 'pool3')

			conv4_1 = self._construct_conv_layer(pool3, 'conv4_1')
			conv4_2 = self._construct_conv_layer(conv4_1, 'conv4_2')
			conv4_3 = self._construct_conv_layer(conv4_2, 'conv4_3')
			conv4_4 = self._construct_conv_layer(conv4_3, 'conv4_4')
			pool4 = self._max_pool(conv4_4, 'pool4')

			conv5_1 = self._construct_conv_layer(pool4, 'conv5_1')
			conv5_2 = self._construct_conv_layer(conv5_1, 'conv5_2')
			conv5_3 = self._construct_conv_layer(conv5_2, 'conv5_3')
			conv5_4 = self._construct_conv_layer(conv5_3, 'conv5_4')
			pool5 = self._max_pool(conv5_4, 'pool5')

			# construct VGG19 network -- full connection layer
			fc6 = self._construct_full_connection_layer(pool5, 'fc6', active = True)
			fc6 = tf.cond(tensor_trainable, lambda: tf.nn.dropout(fc6, 0.5), lambda: fc6)

			fc7 = self._construct_full_connection_layer(fc6, 'fc7', active = True)
			fc7 = tf.cond(tensor_trainable, lambda: tf.nn.dropout(fc7, 0.5), lambda: fc7)

			fc8 = self._construct_full_connection_layer(fc7, 'fc8', active = False)

			# prediction op
			predict = tf.nn.softmax(fc8, name = 'predict')

			return predict, conv5_4

	def _construct_conv_layer(self, input_layer, layer_name):
		assert layer_name in _VGG19_NETWORK
		conv_config = _VGG19_NETWORK[layer_name]

		with tf.variable_scope(layer_name):
			if self.initialized_parameter_dict and layer_name in self.initialized_parameter_dict:
				init_weight = tf.constant_initializer(self.initialized_parameter_dict[layer_name][_WEIGHT_INDEX])
				init_bias = tf.constant_initializer(self.initialized_parameter_dict[layer_name][_BIAS_INDEX])
				# print 'conv initialize from model'
			else:
				init_weight = tf.truncated_normal_initializer(mean = 0.0, stddev = 0.001, dtype = tf.float32)
				init_bias = tf.zeros_initializer(dtype = tf.float32)

			filter_shape = [conv_config[0], conv_config[0], input_layer.get_shape()[3], conv_config[1]]
			weight = tf.get_variable(
				name = layer_name + '_weight',
				shape = filter_shape,
				initializer = init_weight,
				regularizer = tf.contrib.layers.l2_regularizer(_REGULAR_FACTOR))
			bias = tf.get_variable(
				name = layer_name + '_bias',
				shape = [conv_config[1]],
				initializer = init_bias,
				regularizer = None)

			conv = tf.nn.conv2d(input_layer, weight, _CONV_KERNEL_STRIDES, padding = 'SAME')
			active = tf.nn.relu(tf.nn.bias_add(conv, bias))

			return active

	def _construct_full_connection_layer(self, input_layer, layer_name, active = True):
		assert layer_name in _VGG19_NETWORK
		fc_config = _VGG19_NETWORK[layer_name]

		input_dimension = 1
		for dim in input_layer.get_shape().as_list()[1:]:
			input_dimension *= dim

		with tf.variable_scope(layer_name):
			if self.initialized_parameter_dict and layer_name in self.initialized_parameter_dict:
				init_weight = tf.constant_initializer(self.initialized_parameter_dict[layer_name][_WEIGHT_INDEX])
				init_bias = tf.constant_initializer(self.initialized_parameter_dict[layer_name][_BIAS_INDEX])
			else:
				init_weight = tf.truncated_normal_initializer(mean = 0.0, stddev = 0.001, dtype = tf.float32)
				init_bias = tf.zeros_initializer(dtype = tf.float32)
			weight = tf.get_variable(
				name = layer_name + '_weight',
				shape = [input_dimension, fc_config[0]],
				initializer = init_weight,
				regularizer = tf.contrib.layers.l2_regularizer(_REGULAR_FACTOR))
			bias = tf.get_variable(
				name = layer_name + '_bias',
				shape = [fc_config[0]],
				initializer = init_bias,
				regularizer = None)

			reshape_input = tf.reshape(input_layer, [-1, input_dimension])
			if active:
				return tf.nn.relu(tf.nn.bias_add(tf.matmul(reshape_input, weight), bias))
			return tf.nn.bias_add(tf.matmul(reshape_input, weight), bias)

	def _max_pool(self, input_layer, name):
		return tf.nn.max_pool(input_layer, ksize = _MAX_POOL_KSIZE, strides = _MAX_POOL_STRIDES, padding = 'SAME', name = name)
