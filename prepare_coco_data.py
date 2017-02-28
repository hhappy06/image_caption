import os
import math
import numpy as np
import pandas as pd
import cPickle as pickle

from pycocotools.coco import COCO
from model.word2vec import Word2Vec

_TRAIN_IMAGE_DIR_ = ''
_TRAIN_CAPTION_FILE_ = ''
_TRAIN_ANNOTATION_FILE_ = ''

_MAX_SENTENCE_LENGTH_ = 
_VOCAB_SIZE_ = 
_EMBEDDING_DIM_ = 
_WORD2VEC_FILE_PATH_ = ''

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

	coco.filter_by_words(word_table.all_words())

	captions = [coco.anns[ann_id]['caption'] for ann_id in coco.anns]
	symbolize_sentence = []
	masks = []
	for cap in captions:
		tem_symb, temp_mask = word2vec.symbolize_sentence(cap)
		symbolize_sentence.append(tem_symb)
		masks.append(temp_mask)

    image_ids = [coco.anns[ann_id]['image_id'] for ann_id in coco.anns]
    image_files = [os.path.join(_TRAIN_IMAGE_DIR_, coco.imgs[img_id]['file_name']) for img_id in image_ids]
    csv_filelist = pd.DataFrame({'image_id': image_ids, 'image_file': image_files, 'symbolize_sentence': symbolize_sentence, 'mask':masks, 'caption': captions})
    csv_filelist.to_csv(_TRAIN_ANNOTATION_FILE_)

if __name__ == '__main__':
	process_coco_data()