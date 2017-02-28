import os, sys
import tensorflow as tf
import numpy as np
import time
from model.image_caption import ImageCaption
from input.line_parser.line_parser import ImageParser
from input.data_reader import read_data
from loss.image_caption_loss import ImageCaptionLoss
from train_op import train_opt
from model.word2vec import Word2Vec

_Z_DIM_ = 10
_Z_CONT_DIM = 2
_N_CLASS_ = 10

_BATCH_SIZE_ = 64
_EPOCH_ = 10
_TRAINING_SET_SIZE_ = 60000
_DATA_DIR_ = './data/mscoco/train_images'
_CSVFILE_ = ['./data/mscoco/train_images/file_list']
_WORD2VEC_FILE_PATH_ = ''

_OUTPUT_INFO_FREQUENCE_ = 100
_SAVE_MODEL_FREQUENCE_ = 100

line_parser = ImageParser()
image_caption_loss = ImageCaptionLoss()

def train():
	with tf.Graph().as_default():
		# input
		images, sentences, masks = read_data(_CSVFILE_, line_parser = line_parser, data_dir = _DATA_DIR_, batch_size = _BATCH_SIZE_)

		# word2vec dictionary
		word2vec = Word2Vec()
		word2vec.load_word2vec(_WORD2VEC_FILE_PATH_)
		init_embedding = word2vec.get_word2vec_dic()
		embedding_dictionary = tf.get_variable(
			name = 'embedding_dictionary',
			shape = init_embedding.shape,
			initializer = tf.constant_initializer(init_embedding))
		# model 
		image_caption = ImageCaption('image_caption')
		output_logit, output_prob = image_caption.inference(images, sentences, embedding_dictionary, trainable = True, initialized_vgg_parameter_file = None)
		model_loss = image_caption_loss.loss(output_logit, sentences, masks)

		# summary
		sum_model_loss = tf.summary.scalar('model_loss', model_loss)

		# opt
		trainable_vars = tf.trainable_variables()
		model_opt = train_opt(model_loss, trainable_vars)
		
		# initialize variable
		init_op = tf.global_variables_initializer()
		saver = tf.train.Saver(tf.global_variables())

		session = tf.Session()
		file_writer = tf.summary.FileWriter('./logs', session.graph)
		session.run(init_op)

		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=session, coord=coord)

		print 'InfoGAN training starts...'
		sys.stdout.flush()
		counter = 0
		max_steps = int(_TRAINING_SET_SIZE_ / _BATCH_SIZE_)
		for epoch in xrange(_EPOCH_):
			for step in xrange(max_steps):

				_, summary_str, error_d_loss = session.run([model_opt, sum_model_loss, model_loss])
				file_writer.add_summary(summary_str, counter)
				file_writer.flush()
				counter += 1

				if counter % _OUTPUT_INFO_FREQUENCE_ == 0:
					print 'step: (%d, %d), loss: %f'%(epoch, step, error_d_loss)
					sys.stdout.flush()

				if counter % _SAVE_MODEL_FREQUENCE_ == 0:
					saver.save(session, './checkpoint/image_caption', global_step = counter)

		print 'training done!'
		file_writer.close()
		coord.request_stop()
		coord.join(threads)
		session.close()

if __name__ == '__main__':
	train()
