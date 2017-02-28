from __future__ import absolute_import

from input.line_parser.line_parser import ImageParser
import numpy as np
import tensorflow as tf

_NUM_THREAD_ = 4
_MIN_AFTER_DEQUEUE_ = 20

def read_data(file_list, line_parser = ImageParser(), data_dir = '', num_epoch = None, batch_size = 1):
	file_name_queue = tf.train.string_input_producer(file_list, num_epoch)
	line_reader = tf.TextLineReader()
	key, value = line_reader.read(file_name_queue)
	image, sentence, mask = line_parser.parse(value, data_dir)
	capacity = _MIN_AFTER_DEQUEUE_ + batch_size * _NUM_THREAD_
	batch_image, batch_sentence, batch_mask = tf.train.shuffle_batch(
		[image, sentence, mask],
		batch_size = batch_size,
		capacity = capacity,
		min_after_dequeue = _MIN_AFTER_DEQUEUE_,
		num_threads = _NUM_THREAD_)
	return batch_image, batch_sentence, batch_mask