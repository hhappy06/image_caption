from __future__ import absolute_import
import numpy as np
import tensorflow as tf
from model.vgg19 import VGG19

# convolution/pool stride
_CONV_KERNEL_STRIDES_ = [1, 2, 2, 1]
_DECONV_KERNEL_STRIDES_ = [1, 2, 2, 1]
_REGULAR_FACTOR_ = 1.0e-4

_RNN_HIDDEN_NUMER_ = 512
_MLP_LAYER_NUMBER_ = 2
_BATCH_NORM_EPSILON_ =1e-5
_BATCH_NROM_MOMENTUM_ = 0.9

_LAST_FC_DIMENSION_ = 1000

def _construct_conv_layer(input_layer, output_dim, kernel_size = 3, stddev = 0.02, name = 'conv2d'):
	with tf.variable_scope(name):
		init_weight = tf.truncated_normal_initializer(mean = 0.0, stddev = stddev, dtype = tf.float32)
		filter_size = [kernel_size, kernel_size, input_layer.get_shape()[-1], output_dim]
		weight = tf.get_variable(
			name = name + 'weight',
			shape = filter_size,
			initializer = init_weight,
			regularizer = tf.contrib.layers.l2_regularizer(_REGULAR_FACTOR_))
		bias = tf.get_variable(
			name = name + 'bias',
			shape = [output_dim],
			initializer = tf.constant_initializer(0.0))
		conv = tf.nn.conv2d(input_layer, weight, _CONV_KERNEL_STRIDES_, padding = 'SAME')
		conv = tf.nn.bias_add(conv, bias)
		return conv

def _construct_full_connection_layer(input_layer, output_dim, stddev = 0.02, name = 'fc'):
	# calculate input_layer dimension and reshape to batch * dimension
	input_dimension = 1
	for dim in input_layer.get_shape().as_list()[1:]:
		input_dimension *= dim

	with tf.variable_scope(name):
		init_weight = tf.truncated_normal_initializer(mean = 0.0, stddev = stddev, dtype = tf.float32)
		filter_size = [input_dimension, output_dim]
		weight = tf.get_variable(
			name = name + 'weight',
			shape = filter_size,
			initializer = init_weight,
			regularizer = tf.contrib.layers.l2_regularizer(_REGULAR_FACTOR_))
		bias = tf.get_variable(
			name = name + 'bias',
			shape = [output_dim],
			initializer = tf.constant_initializer(0.0))
		input_layer_reshape = tf.reshape(input_layer, [-1, input_dimension])
		fc = tf.matmul(input_layer_reshape, weight)
		tc = tf.nn.bias_add(fc, bias)
		return fc

#defination of image-caption network
class ImageCaption:
	def __init__(self, name = 'ImageCaption'):
		self.name = name

	def inference(self, input_images, sentences, embedding_dictionary, trainable, initialized_vgg_parameter_file = None):
		with tf.variable_scope(self.name):
			# extract image context feature
			tensor_trainable = tf.constant(trainable)
			vgg19 = VGG19('vgg19')
			vgg19_predict, vgg19_tensor_trainable, vgg19_context = vgg19.inference(input_images, tensor_trainable, initialized_vgg_parameter_file)

			vgg19_context_shape = vgg19_context.get_shape().as_list()
			batch_size = vgg19_context_shape[0]
			vgg19_context_num = vgg19_context_shape[1] * vgg19_context_shape[2]
			vgg19_context_dim = vgg19_context_shape[3]
			vgg19_context_reshape = tf.reshape(vgg19_context, [-1, vgg19_context_num, vgg19_context_dim])
			vgg19_context_reshape_mean = tf.reduce_mean(vgg19_context_reshape, 1)

			init_memory = vgg19_context_reshape_mean
			for i in xrange(_MLP_LAYER_NUMBER_):
				init_memory = _construct_full_connection_layer(init_memory, _RNN_HIDDEN_NUMER_, name = 'init_memory_fc' + str(i))
				init_memory = tf.contrib.layers.batch_norm(init_memory,
								decay = self.momentum, 
								updates_collections = None,
								epsilon = self.epsilon,
								scale = True,
								is_training = trainable,
								scope = 'init_memory_bn' + str(i))

			init_lstm_output = vgg19_context_reshape_mean
			for i in xrange(_MLP_LAYER_NUMBER_):
				init_lstm_output = _construct_full_connection_layer(init_lstm_output, _RNN_HIDDEN_NUMER_, name = 'init_hidden_state_fc' + str(i))
				init_lstm_output = tf.contrib.layers.batch_norm(init_lstm_output,
								decay = self.momentum, 
								updates_collections = None,
								epsilon = self.epsilon,
								scale = True,
								is_training = trainable,
								scope = 'init_lstm_output_bn' + str(i))

			lstm_state = tf.contrib.rnn.LSTMStateTuple(init_memory, init_lstm_output)
			lstm = tf.contrib.rnn.LSTMCell(_RNN_HIDDEN_NUMER_, initializer=tf.random_normal_initializer(stddev=0.03))

			vgg19_context_flat = tf.reshape(vgg19_context_reshape, [-1, vgg19_context_dim])

			max_sentence_length = sentences.get_shape().as_list()[-1]
			dim_embed = embedding_dictionary.get_shape().as_list()[-1]
			word_number = embedding_dictionary.get_shape().as_list()[0]
			tensor_output = []
			tensor_output_prob = []
			for i in xrange(max_sentence_length):
				# attention mechanism
				context_encode1 = _construct_full_connection_layer(vgg19_context_flat, vgg19_context_dim, name = 'att_fc11')
				context_encode1 = tf.nn.relu(context_encode1)
				context_encode1 = tf.contrib.layers.batch_norm(context_encode1,
								decay = self.momentum, 
								updates_collections = None,
								epsilon = self.epsilon,
								scale = True,
								is_training = trainable,
								scope = 'att_bn11' + str(i))

				context_encode2 = _construct_full_connection_layer(lstm_state, vgg19_context_dim, name = 'att_fc21')
				context_encode2 = tf.nn.relu(context_encode2)
				context_encode2 = tf.contrib.layers.batch_norm(context_encode2,
								decay = self.momentum, 
								updates_collections = None,
								epsilon = self.epsilon,
								scale = True,
								is_training = trainable,
								scope = 'att_bn21' + str(i))
				context_encode2 = tf.tile(tf.expand_dims(context_encode2, 1), [1, vgg19_context_num, 1])
				context_encode2 = tf.reshape(context_encode2, [-1, vgg19_context_dim])

				context_encode = tf.relu(context_encode1 + context_encode2)
				context_encode = tf.cond(tensor_trainable, lambda: tf.nn.dropout(context_encode, 0.5), lambda: context_encode)

				attention = _construct_full_connection_layer(context_encode, 1, name = 'att_1')
				attention = tf.nn.relu(attention)
				attention = tf.reshape(attention, [-1, vgg19_context_num])
				attention = tf.nn.softmax(attention)

				if i == 0:
					word_emb = tf.zeros([batch_size, dim_embed])
					weighted_context = tf.identity(vgg19_context_reshape_mean)
				else:
					word_emb = tf.cond(is_train, lambda: tf.nn.embedding_lookup(embedding_dictionary, sentences[:, i-1]), lambda: word_emb)
					weighted_context = tf.reduce_sum(vgg19_context_reshape * tf.expand_dims(attention, 2), 1)

				lstm_output, lstm_state = lstm(tf.concat(1, [weighted_context, word_emb]), lstm_state)
				feature_concate = tf.concat(1, [lstm_output, weighted_context, word_emb])
				output0 = _construct_full_connection_layer(feature_concate, _LAST_FC_DIMENSION_, name = 'output_fc1')
				output0 = tf.nn.tanh(output0)
				output0 = tf.cond(tensor_trainable, lambda: tf.nn.dropout(output0, 0.5), lambda: output0)

				output = _construct_full_connection_layer(output0, word_number)
				prob = tf.nn.softmax(output)

				tensor_output.append(output)
				tensor_output_prob.append(prob)

				max_prob_word = tf.argmax(output, 1)
				word_emb = tf.cond(tensor_trainable, lambda: word_emb, lambda: tf.nn.embedding_lookup(emb_w, max_prob_word))          
				tf.get_variable_scope().reuse_variables()

			tensor_output = tf.pack(tensor_output, axis = 1)
			tensor_output_prob = tf.pack(tensor_output_prob, axis = 1)

		return tensor_output, tensor_output_prob
