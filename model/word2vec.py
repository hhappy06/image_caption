from __future__ import absolute_import
import os
import numpy as np
import cPickle as pickle

class Word2Vec:
	def __init__(self, vocab_size = 5000, embed_dim = 256, max_sent_len = 30):
		self.vocab_size = vocab_size
		self.embed_dim = embed_dim
		self.max_sent_len = max_sent_len

		self.word2vec = {}
		self.idx2word = []
		self.word2idx = {}

	def build_word2vec(self, sentences):
		word_count = {}
		for sent in sentences:
			for word in sent.lower().split(' '):
				 word_count[word] = word_count.get(word, 0) + 1

		sorted_word_count = sorted(list(word_count.items()), key=lambda x: x[1], reverse=True) 
		self.vocab_size = min(len(sorted_word_count), self.vocab_size)
        
		for idx in range(self.vocab_size):
			word, _ = sorted_word_count[idx]
			self.idx2word.append(word)
			self.word2idx[word] = idx
			self.word2vec[word] = 0.01 * np.random.randn(self.embed_dim)

	def get_word2vec_dic(self):
		return self.word2vec

	def get_vocab_size(self):
		return self.vocab_size

	def get_embedding_dim(self):
		return self.embed_dim

	def get_embedding_table(self):
		embedding_table = []
		print type(self.word2vec)
		for word in self.idx2word:
			print self.word2vec[word]
			exit()
			embedding_table.append(self.word2vec[word])
		return np.array(embedding_table)

	def save_word2vec(self, path):
		with open(path, 'w') as output2file:
			 pickle.dump([self.idx2word, self.word2idx, self.word2vec, self.vocab_size, self.embed_dim, self.max_sent_len], output2file)

	def load_word2vec(self, path):
		with open(path, 'r') as file2input:
			self.idx2word, self.word2idx, self.word2vec, self.vocab_size, self.embed_dim, self.max_sent_len = pickle.load(file2input)

	def symbolize_sentence(self, sentence):
		indices = np.zeros(self.max_sent_len).astype(np.int32)
		masks = np.zeros(self.max_sent_len)
		word_idx = np.array([self.word2idx[w] for w in sentence.lower().split(' ')])
		indices[:len(word_idx)] = word_idx
		masks[:len(word_idx)] = 1.0
		return indices, masks

	def indices_to_sentence(self, indices):
		words = [self.idx2word[idx] for idx in indices]
		if words[-1] != '.':
			words.append('.')
		punctuation = np.argmax(np.array(words) == '.') + 1
		sentence = words[:punctuation]
		sentence = ' '.join(words)
		sentence = sentence.replace(' ,', ',')
		sentence = sentence.replace(' ;', ';')
		sentence = sentence.replace(' :', ':')
		sentence = sentence.replace(' .', '.')
		return sentence

	def get_all_words(self):
		return set(self.word2idx.keys())


