from __future__ import absolute_import
import tensorflow as tf

class ImageCaptionLoss:
	def loss(self, output_logit, sentences, masks, ):
		word_number = output_logit.get_shape().as_list()[-1]
		output_logit_reshape = tf.reshape(output_logit, [-1, word_number])
		sentences_reshape = tf.reshape(sentences, [-1, 1])
		masks_reshape = tf.reshape(sentences, [-1, 1])
		cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(output_logit_reshape, sentences_reshape)
		cross_entropy = cross_entropy * masks_reshape
		model_loss = tf.reduce_mean(cross_entropy)
		return model_loss