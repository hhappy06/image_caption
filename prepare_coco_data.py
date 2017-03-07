import os
import math
import numpy as np
import pandas as pd
import cPickle as pickle

from model.word2vec import Word2Vec
from pycocotools.coco import COCO

_TRAIN_CAPTION_FILE_ = './data/mscoco/annotations/captions_val2014.json'
_TRAIN_ANNOTATION_FILE_ = './data/mscoco/annotations/anna.csv'

_MAX_SENTENCE_LENGTH_ = 20
_VOCAB_SIZE_ = 1000
_EMBEDDING_DIM_ = 256
_WORD2VEC_FILE_PATH_ = './word2vec_table/word2vec.pickle'

def process_coco_data():
	coco = COCO(_TRAIN_CAPTION_FILE_)
	coco.filter_by_cap_len(_MAX_SENTENCE_LENGTH_)

	# vocab_size, embed_dim, max_sent_len
	word2vec = Word2Vec(_VOCAB_SIZE_, _EMBEDDING_DIM_, _MAX_SENTENCE_LENGTH_)
	if not os.path.exists(_WORD2VEC_FILE_PATH_):
		word2vec.build_word2vec(coco.all_captions())
		word2vec.save_word2vec(_WORD2VEC_FILE_PATH_)
	else:
		word2vec.load_word2vec(_WORD2VEC_FILE_PATH_)

	coco.filter_by_words(word2vec.get_all_words())

	captions = [coco.anns[ann_id]['caption'] for ann_id in coco.anns]
	symbolize_sentence = []
	masks = []
	print 'symbolize sentence... total %d'%(len(captions))
	for cap in captions:
		tem_symb, temp_mask = word2vec.symbolize_sentence(cap)
		tem_symb_sentence = ' '.join([str(v) for v in tem_symb])
		temp_mask_sentence = ' '.join([str(v) for v in temp_mask])
		symbolize_sentence.append(tem_symb_sentence)
		masks.append(temp_mask_sentence)

	image_ids = [coco.anns[ann_id]['image_id'] for ann_id in coco.anns]
	image_files = [coco.imgs[img_id]['file_name'] for img_id in image_ids]
	# csv_filelist = pd.DataFrame({'image_id': image_ids, 'image_file': image_files, 'symbolize_sentence': symbolize_sentence, 'mask':masks, 'caption': captions})
	# csv_filelist.to_csv(_TRAIN_ANNOTATION_FILE_)
	with open(_TRAIN_ANNOTATION_FILE_) as output2file:
		for idx in xrange(len(image_ids)):
			one_row = ','.join([image_ids[idx], image_files[idx], symbolize_sentence[idx], masks[idx]]) + '\n'
			output2file.write(one_row)

if __name__ == '__main__':
	process_coco_data()