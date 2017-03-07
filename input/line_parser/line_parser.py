from __future__ import absolute_import

import numpy as np
import tensorflow as tf

_IMAGE_HEIGHT_ORG_ = 28
_IMAGE_WIDTH_ORG_ = 28
_IMAGE_DEPTH_ORG_ = 3

_IMAGE_SIZE_ = 256

class ImageParser():
	def parse(self, value, data_dir = ''):
		image_id, image_name, image_symb_cap, cap_mask = tf.decode_csv(value, [[''], [''], [''], ['']])
		if data_dir:
			image_name = data_dir + '/' + image_name
		png = tf.read_file(image_name)
		# print file_name
		image = tf.image.decode_png(png, channels = 3)
		image = tf.cast(image, tf.float32)
		resize_image = tf.image.resize_images(image, [_IMAGE_SIZE_, _IMAGE_SIZE_])
		sentence = tf.cast(tf.string_split(image_symb_cap, ' '), tf.float32)
		mask = tf.cast(tf.string_split(cap_mask, ' '), tf.float32)

		return resize_image, sentence, mask
